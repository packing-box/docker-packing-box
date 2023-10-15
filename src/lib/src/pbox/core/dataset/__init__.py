# -*- coding: UTF-8 -*-
from tinyscript import colored, hashlib, json, random, time
from tinyscript.helpers import confirm, human_readable_size, lazy_load_module, slugify, Path, TempPath
from tinyscript.report import *

from .plot import *
from ..executable import *
from ..pipeline import *
from ...helpers import *

lazy_load_module("alterations", "pbox.core.executable")
lazy_load_module("packer", "pbox.core.items", "item_packer")


__all__ = ["Dataset", "FilelessDataset"]


def backup(f):
    """ Simple method decorator for making a backup of the dataset. """
    def _wrapper(s, *a, **kw):
        if config['backup_copies'] > 0:
            s.backup = s
        return f(s, *a, **kw)
    return _wrapper


class Dataset(AbstractEntity):
    """ Folder structure:
    
    [name]
      +-- files
      |     +-- {executables, renamed to their SHA256 hashes}
      +-- data.csv            # metadata and labels of the executable
      +-- metadata.json       # simple statistics about the dataset
      +-- (features.json)     # if features have been computed, dictionary of selected features and their descriptions
      +-- (alterations.json)  # if the dataset was altered, this contains the hashes of the altered executables with the
                              #  alterations applied
    """
    def __init__(self, name="dataset", source_dir=None, load=True, name_check=False, **kw):
        self._files = getattr(self.__class__, "_files", True)
        if name_check:
            config.check(Path(name).basename)
        self.path = Path(config['datasets'].joinpath(name), create=load).absolute()
        self.sources = source_dir or PACKING_BOX_SOURCES
        if isinstance(self.sources, list):
            self.sources = {'All': [str(x) for x in self.sources]}
        for _, sources in self.sources.items():
            for source in sources[:]:
                s = Path(source, expand=True)
                if not s.exists() or not s.is_dir():
                    sources.remove(source)
        if load:
            self._load()
        self.formats = getattr(self, "_metadata", {}).get('formats', [])
        self.__change = False
        self.__limit = 20
        self.__per_format = False
    
    def __delitem__(self, executable):
        """ Remove an executable (by its real name or hash) from the dataset. """
        self.__change, df = True, self._data
        h = executable
        # first, ensure we handle the hash name (not the real one)
        try:
            h = df.loc[df['realpath'] == h, 'hash'].iloc[0]
        except:
            if isinstance(h, Executable):
                h = h.hash
        if len(df) > 0:
            self.logger.debug("removing %s..." % h)
            self._data = df[df.hash != h]
        if self._files:
            self.files.joinpath(h).remove(error=False)
            try:
                ext = Path(df.loc[df['hash'] == h, 'realpath'].iloc[0]).extension
                self.files.joinpath(h + ext).remove(error=False)
            except:
                pass
    
    def __eq__(self, dataset):
        """ Custom equality function. """
        ds = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        return self._metadata == ds._metadata and self._data.equals(ds._data) and \
               (not self._files or list(self.files.listdir()) == list(ds.files.listdir()))
    
    def __getitem__(self, hash):
        """ Get a data row related to the given hash from the dataset. """
        try:
            h, with_headers = hash
        except ValueError:
            h, with_headers = hash, False
        if len(self._data) == 0:
            raise KeyError(h)
        try:
            row = self._data[self._data.hash == h].iloc[0]
            return row.to_dict() if with_headers else row.to_list()
        except IndexError:
            raise KeyError(h)
    
    def __hash__(self):
        """ Custom object hashing function. """
        return int.from_bytes(hashlib.md5(self.name.encode()).digest(), "little")
    
    def __iter__(self):
        """ Iterate over dataset's sample executables, computing the features and filling their dictionary too if self
             is a FilelessDataset. """
        if not self._files and not hasattr(self, "_features"):
            self._features = {}
        for row in self._data.itertuples():
            exe = Executable(row, dataset=self)
            exe._row = row
            if not self._files:
                self._features.update(exe.features)
            yield exe
    
    def __len__(self):
        """ Get dataset's length. """
        return len(self._data.index)
    
    def __setattr__(self, name, value):
        # auto-expand the formats attribute into a private one
        if name == "formats":
            self._formats_exp = expand_formats(*value)
        super(Dataset, self).__setattr__(name, value)
    
    def __setitem__(self, executable, label):
        """ Add an executable based on its real name to the dataset.
        
        :param executable: either the path to the executable or its Executable instance
        :param label:      either the text label of the given executable or its dictionary of data
        """
        try:
            label, update = label
        except (TypeError, ValueError):  # TypeError occurs when label is None
            label, update = label, False
        self.__change, l = True, self.logger
        df, e = self._data, executable
        e = e if isinstance(e, Executable) else Executable(e, dataset=self)
        # case (1) 'label' is a dictionary with the executable's attributes, i.e. from another dataset
        if isinstance(label, dict):
            d = label
            # get metadata values from the input dictionary
            for k, v in d.items():
                if k in Executable.FIELDS:
                    setattr(e, k, v)
            # then ensure we compute the remaining values
            for k in ['hash'] + Executable.FIELDS:
                if k not in d:
                    d[k] = getattr(e, k)
        # case (2) 'label' is the label value
        else:
            d = e.metadata
            d['hash'] = e.hash
            d['label'] = e.label = label
        if self._files:
            e.copy()
        self._metadata.setdefault('formats', [])
        if e.format not in self._metadata['formats']:
            self._metadata['formats'].append(e.format)
        if len(df) > 0 and e.hash in df.hash.values:
            row = df.loc[df['hash'] == e.hash]
            lbl = row['label'].iloc[0]
            # consider updating when:
            #  (a) hash already exists but is not packed => can pack it
            #  (b) new fields are added, e.g. when converting to FilelessDataset (features are computed)
            if str(lbl) == "nan" or update:
                l.debug("updating %s..." % e.hash)
                df.loc[df['hash'] == e.hash, d.keys()] = d.values()
            else:
                l.debug("discarding %s%s..." % (e.hash, ["", " (already in dataset)"][lbl == d['label']]))
        else:
            l.debug("adding %s..." % e.hash)
            self._data = pd.concat([df, pd.DataFrame.from_records([d])], ignore_index=True)
    
    def _compute_all_features(self):
        """ Convenience function for computing the self._data pandas.DataFrame containing the feature values. """
        self.logger.info("Computing features..." if self._files else "Loading features...")
        with progress_bar() as p:
            for exe in p.track(self):
                d = self._compute_features(exe)
                if self._files:
                    self[exe.hash] = (d, True)  # True: force updating the row
    
    def _compute_features(self, exe):
        """ Compute the features for a single Executable instance. """
        exe = Executable(exe, dataset=self, force=True)
        d = self[exe.hash, True]  # retrieve executable's record as a dictionary
        d.update(exe.data)        # be sure to include the features
        if not hasattr(self, "_features"):
            self._features = {}
            self._features.update(exe.features)
        return d
    
    def _copy(self, path):
        """ Copy the current dataset to a given destination. """
        self.logger.debug("copying dataset '%s' to %s" % (self.basename, path))
        self.path.copy(path)
    
    def _load(self):
        """ Load dataset's associated files or create them. """
        l = self.logger
        if not self.path.exists():
            l.debug("creating path %s..." % self.path)
            self.path.mkdir(exist_ok=True)
        try:
            if len(self) > 0:
                l.debug("loading dataset '%s'..." % self.basename)
        except AttributeError:  # self._data does not exist yet
            pass
        l.debug("dataset: type=%s (class: %s), name=%s" % \
                (["fileless", "normal"][self._files], self.__class__.__name__, self.name))
        if self._files:
            self.files.mkdir(exist_ok=True)
        for n in ["alterations", "data", "metadata"] + [["features"], []][self._files]:
            p = self.path.joinpath(n + [".json", ".csv"][n == "data"])
            if n == "data":
                try:
                    self._data = pd.read_csv(str(p), sep=";", parse_dates=['ctime', 'mtime'])
                except (OSError, KeyError):
                    self._data = pd.DataFrame()
            elif p.exists():
                with p.open() as f:
                    setattr(self, "_" + n, json.load(f))
            else:
                setattr(self, "_" + n, {})
    
    def _purge(self, backup=False, **kw):
        """ Purge the current dataset, including its backups. """
        self.logger.debug("purging %s%s..." % (self.path, ["", "'s backups"][backup]))
        if not backup:
            self._remove()
        # also recursively purge the backups
        try:
            self.backup._purge()
        except AttributeError:
            pass
    
    def _remove(self):
        """ Remove the current dataset. """
        self.logger.debug("removing dataset '%s'..." % self.basename)
        self.path.remove(error=False)
    
    def _save(self):
        """ Save dataset's state to JSON files. """
        l = self.logger
        if not self.__change:
            return
        if len(self) == 0 and not Dataset.check(self.basename):
            self._remove()
            return
        l.debug("saving dataset '%s'..." % self.basename)
        self._metadata['formats'] = sorted(collapse_formats(*self._metadata['formats']))
        try:
            self._metadata['counts'] = {k: v for k, v in self._data.label.value_counts().to_dict().items()}
        except AttributeError:
            l.warning("No label found")
            self._remove()
            return
        self._metadata['executables'] = len(self)
        for n in ["alterations", "data", "metadata"] + [["features"], []][self._files]:
            if n == "data":
                self._data = self._data.sort_values("hash")
                fields = ["hash"] + Executable.FIELDS + ["label"]
                fnames = [h for h in self._data.columns if h not in fields + ["Index"]]
                c = fields[:-1] + fnames + [fields[-1]]
                self._data.to_csv(str(self.path.joinpath("data.csv")), sep=";", columns=c, index=False, header=True)
            else:
                attr = getattr(self, "_" + n)
                if len(attr) > 0:
                    with self.path.joinpath(n + ".json").open('w+') as f:
                        json.dump(attr, f, indent=2)
        self.__change = False
    
    def _walk(self, walk_all=False, sources=None, silent=False):
        """ Walk the sources for random in-scope executables. """
        l = self.logger
        [l.info, l.debug][silent]("Searching for executables...")
        m, candidates, packers = 0, [], [p.name for p in item_packer.Packer.registry]
        for cat, srcs in (sources or self.sources).items():
            if all(c not in expand_formats(cat) for c in self._formats_exp):
                continue
            for src in srcs:
                for exe in Path(src, expand=True).walk(filter_func=lambda x: x.is_file(), sort=False):
                    exe = Executable(exe, dataset=self)
                    if exe.format is None or exe.format not in self._formats_exp or exe.stem in packers:
                        continue  # ignore unrelated files and packers themselves
                    if walk_all:
                        yield exe
                    else:
                        candidates.append(exe)
        if len(candidates) > 0 and not walk_all:
            random.shuffle(candidates)
            for exe in candidates:
                yield exe
    
    @backup
    def alter(self, new_name=None, packed_only=False, percentage=1., query=None, **kw):
        """ Alter executables with some given alterations. """
        l = self.logger
        if not self._files:
            l.warning("Alterations work only if the files are available")
            return
        if new_name is not None:
            ds = Dataset(new_name)
            ds.merge(self.path.basename, silent=True, **kw)
            ds.alter(packed_only, percentage, query, **kw)
            return
        limit = 0
        # filter out already altered samples first
        altered_h = [h for hlst in self._alterations.values() for h in hlst]
        df = self._data[~self._data.hash.isin(altered_h)]
        if packed_only:
            df = df[~df.label.isin([NOT_LABELLED, NOT_PACKED])]
            if len(df) == 0:
                l.warning("Nothing more to alter")
                return
            l.info("Altering packed samples of the dataset...")
        elif percentage is not None:
            # keep previous alteration percentage into account
            a = self._metadata.get('altered', .0)
            p = min(1. - a, percentage)
            p_ = round(p * 100, 0)
            if p != percentage:
                if p <= .0:
                    l.warning("Nothing more to alter")
                    return
                else:
                    l.warning("Setting alterations percentage to %d" % p_)
            l.info("Altering %d%% of the dataset..." % p_)
            limit = int(round(len(self)*p, 0))
        elif query is not None:
            l.info("Altering the selected records of the dataset...")
        for e in filter_data_iter(df, query, limit, logger=self.logger):
            exe = Executable(dataset=self, hash=e.hash)
            exe.chmod(0o600)
            for m in alterations.Alterations(exe):
                self._alterations.setdefault(m, [])
                self._alterations[m].append(e.hash)
            exe.chmod(0o400)
        self._metadata['altered'] = sum(1 for x in set(h for hl in self._alterations.values() for h in hl)) / len(self)
        self.__change = True
        self._save()
    
    def browse(self, query=None, no_feature=False, **kw):
        if not no_feature:
            self._compute_all_features()
        with data_to_temp_file(filter_data(self._data, query, logger=self.logger), prefix="dataset-browsing-") as tmp:
            edit_file(tmp, logger=self.logger)
    
    @backup
    def convert(self, new_name=None, **kw):
        """ Convert a dataset with files to a dataset without files. """
        l = self.logger
        l_info = getattr(l, ["info", "debug"][kw.get('silent', False)])
        if not self._files:
            l.warning("Already a fileless dataset")
            return
        if new_name is not None and self.name != new_name:
            kw.pop('name', None)
            self.select(new_name, **kw)
            Dataset(new_name, **kw).convert(**kw)
            return
        l_info("Converting to fileless dataset...")
        s1 = self.path.size
        l_info("Size of dataset:     %s" % human_readable_size(s1))
        self.path.joinpath("features.json").write_text("{}")
        self._compute_all_features()
        self._files = False
        l.debug("removing files...")
        self.files.remove(error=False)
        l.debug("removing eventual backups...")
        try:
            self.backup._purge()
        except AttributeError:
            pass
        self._save()
        s2 = self.path.size
        l_info("Size of new dataset: %s (compression factor: %d)" % (human_readable_size(s2), int(s1/s2)))
    
    def edit(self, **kw):
        """ Edit the data CSV file. """
        l = self.logger
        l.debug("editing dataset's data.csv...")
        edit_file(self.path.joinpath("data.csv").absolute(), logger=l)
    
    def export(self, format=None, output=None, **kw):
        """ Export either packed executables from the dataset to the given output folder or the complete dataset to a
             given format. """
        l = self.logger
        dst = output or self.basename
        if format == "packed-samples":
            if not self._files:
                l.warning("Packed samples can only be exported from a normal dataset (not on a fileless one)")
                return
            dst, n = Path(dst, create=True), kw.get('n', 0)
            lst, tmp = [e for e in self if e.label not in [NOT_PACKED, NOT_LABELLED]], []
            if n > len(lst):
                l.warning("%d packed samples were requested but only %d were found" % (n, len(lst)))
            n = min(n, len(lst))
            l.info("Exporting %d packed executables from %s to '%s'..." % (n, self.basename, dst))
            if 0 < n < len(lst):
                random.shuffle(lst)
            with progress_bar("packed samples") as p:
                for exe in enumerate(lst[:n]):
                    fn = "%s_%s" % (exe.label, Path(exe.realpath).filename)
                    if fn in tmp:
                        l.warning("duplicate '%s'" % fn)
                        continue
                    exe.destination.copy(dst.joinpath(fn))
                    tmp.append(fn)
            return
        if self._files:
            l.info("Computing features...")
            self._compute_all_features()
        fields = ["hash"] + Executable.FIELDS
        fnames = [h for h in self._data.columns if h not in fields + ["label", "Index"]]
        c = fields[:-1] + fnames + ["label"]
        d = self._data[c].values.tolist()
        d.insert(0, c)
        ext = ".%s" % format
        if not dst.endswith(ext):
            dst += ext
        from dsff import DSFF
        if format == "dsff":
            l.info("Exporting dataset %s to '%s'..." % (self.basename, dst))
            with DSFF(self.basename, 'w+') as f:
                f.write(d, self._features, self._metadata)
            Path(self.basename + ext).rename(dst)
        elif format in ["arff", "csv"]:
            l.info("Exporting dataset %s to '%s'..." % (self.basename, dst))
            with DSFF("<memory>") as f:
                f.name = self.basename
                f.write(d, self._features, self._metadata)
                getattr(f, "to_%s" % format)()
            Path(self.basename + ext).rename(dst)
        else:
            raise ValueError("Unknown target format (%s)" % format)
    
    @backup
    def fix(self, **kw):
        """ Make dataset's structure and files match. """
        self.logger.debug("dropping duplicates...")
        self._data = self._data.drop_duplicates()
        if self._files:
            for exe in self.files.listdir(is_exe):
                h = exe.basename
                if Executable(exe).format is None:  # unsupported or bad format (e.g. Bash script)
                    del self[h]
                elif h not in self._data.hash.values:
                    del self[h]
        for exe in self:
            h = exe.hash
            if exe.format is None:
                del self[h]
            elif self._files and not self.files.joinpath(h).exists():
                del self[h]
        self._save()
    
    def ingest(self, folder, labels=None, rename_func=slugify, detect=False, min_samples=0, max_samples=0, merge=False,
               prefix="", exclude=None, **kw):
        """ Ingest every subfolder of a target folder to make it a dataset provided a dictionary of labels. """
        l, p = self.logger, Path(folder)
        name = prefix + rename_func(p.stem)
        kw['silent'] = True
        if merge:
            dataset = Dataset(name, load=False)
            if dataset.exists():
                if confirm("Are you sure you want to overwrite '%s' ?" % dataset.basename):
                    dataset._purge()
                else:
                    return
            dataset._load()
        l.info("Searching for subfolders with samples to be ingested %s..." % \
               ("in the new dataset '%s'" % dataset.basename if merge else "as distinct datasets"))
        for sp in p.walk(filter_func=lambda x: x.is_dir() and all(not y.startswith(".") for y in x.parts[1:])):
            if any(x in (exclude or []) for x in sp.parts[1:]):
                l.debug("%s was excluded by the user" % sp.stem)
                continue
            i, keep = 0, True
            for f in sp.listdir():
                # only select "leaf" subfolder, that is those with only files
                if not f.is_file():
                    keep = False
                    break
                # count the executables in this subfolder, other files (e.g. could be README.md) are not an issue
                if Executable(f).is_valid():
                    i += 1
            # check that we have a subfolder that contains not less and not more executable samples than needed
            if not keep:
                l.debug("%s is not a leaf subfolder" % sp.stem)
                continue
            if min_samples > 0 and i < min_samples:
                l.debug("%s has too few samples" % sp.stem)
                continue
            if max_samples > min_samples and i > max_samples:
                l.debug("%s has too much samples" % sp.stem)
                continue
            l.debug("found a subfolder called %s that has %d executable samples" % (sp.stem, i))
            if not merge:
                name = prefix + rename_func(sp.stem)
                dataset = Dataset(name, load=False)
                if dataset.exists():
                    if confirm("Are you sure you want to overwrite '%s' ?" % dataset.basename):
                        dataset._purge()
                    else:
                        return
                dataset._load()
            l.info("Ingesting subfolder '%s' into %s..." % (sp.stem, name))
            dataset.update([sp, p][merge], labels=labels, detect=detect, **kw)
    
    @backup
    def make(self, n=0, formats=["All"], balance=False, packer=None, pack_all=False, **kw):
        """ Make n new samples in the current dataset among the given binary formats, balanced or not according to
             the number of distinct packers. """
        l, self.formats = self.logger, formats  # this triggers creating self._formats_exp
        # select enabled and non-failing packers among the input list
        packers = [p for p in (packer or item_packer.Packer.registry) if p in item_packer.Packer.registry and \
                                                                         p.check(*self._formats_exp, silent=False)]
        if len(packers) == 0:
            l.critical("No valid packer selected")
            return
        # then restrict dataset's formats to these of the selected packers
        pformats = aggregate_formats(*[tuple(p.formats) for p in packers])
        self.formats = collapse_formats(*[f for f in expand_formats(*formats) if f in pformats])
        sources = []
        for fmt, src in self.sources.items():
            if all(f not in expand_formats(fmt) for f in self._formats_exp):
                continue
            sources.extend(src)
        l.info("Source directories: %s" % ",".join(map(str, set(sources))))
        l.info("Considered formats: %s" % ",".join(self.formats))  # this updates self._formats_exp
        l.info("Selected packers:   %s" % ",".join(["All"] if packer is None else \
                                                   [p.__class__.__name__ for p in packer]))
        self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + sources)))
        if n == 0:
            n = len(list(self._walk(n <= 0, silent=True)))
        # get executables to be randomly packed or not
        n1 = self._metadata.get('executables', 0)
        CBAD, CGOOD = n // 3, n // 3
        i, cbad, cgood = 0, {p: CBAD for p in packers}, {p: CGOOD for p in packers}
        with progress_bar() as progress:
            pbar = progress.add_task("", total=None if n <= 0 else n)
            for exe in self._walk(n <= 0):
                label = short_label = NOT_PACKED
                to_be_packed = pack_all or random.randint(0, len(packers) if balance else 1)
                # check 1: are there already samples enough?
                if i >= n > 0:
                    break
                # check 2: are there working packers remaining?
                if len(packers) == 0:
                    l.critical("No packer left")
                    return
                # check 3: is the selected Executable supported by any of the remaining packers?
                if all(not p._check(exe, silent=True) for p in packers):
                    l.debug("unsupported file (%s)" % exe)
                    continue
                # check 4: was this executable already included in the dataset?
                if len(self._data) > 0 and self._files and exe.destination.exists():
                    l.debug("already in the dataset (%s)" % exe)
                    continue
                l.debug("handling %s..." % exe)
                if to_be_packed:
                    if len(packers) > 1:
                        random.shuffle(packers)
                    dest = exe.copy(extension=True)
                    if dest is None:  # occurs when the copy failed
                        continue
                    dest = dest.absolute()
                    for p in packers[:]:
                        fmt = dest.format
                        dest.chmod(0o700 if getattr(p, "xflag", False) else 0o600)
                        label = p.pack(dest)
                        # means that this kind of executable is not supported by this packer
                        if label is None:
                            continue
                        # means that the executable was packed but modifying the file type
                        if fmt != dest.format and label not in [NOT_LABELLED, NOT_PACKED]:
                            self.logger.debug("resetting %s..." % exe)
                            dest.remove()
                            dest = exe.copy(extension=True).absolute()  # reset the original executable
                            label = NOT_PACKED
                            continue
                        if label == NOT_PACKED or p._bad:
                            # if we reached the limit of GOOD packing occurrences, we consider the packer as GOOD again
                            if cgood[p] <= 0:
                                p._bad, cbad[p] = False, n // 3
                            # but if BAD, we reset the GOOD counter and eventually disable it
                            if p._bad:
                                cbad[p] -= 1      # update BAD counter
                                cgood[p] = CGOOD  # reset GOOD counter ; if still in BAD state, then we need 'cgood'
                                                  #  successful packings to return to the GOOD state
                                if cbad[p] <= 0:  # BAD counter exhausted => disable the packer
                                    l.warning("Disabling %s..." % p.__class__.__name__)
                                    packers.remove(p)
                            # if GOOD and label is None
                            else:
                                cgood[p] -= 1   # update GOOD counter
                                cbad[p] = CBAD  # reset BAD counter
                            label = NOT_PACKED
                            continue
                        # consider short label (e.g. "midgetpack", not "midgetpack[<password>]")
                        short_label = label.split("[")[0]
                        break
                    dest.chmod(0o400)
                    # ensure we did not left the executable name with its hash AND extension behind
                    try:
                        dest.rename(exe.destination)
                    except FileNotFoundError:
                        pass
                if not pack_all or pack_all and label not in [NOT_PACKED, NOT_LABELLED]:
                    self[exe] = short_label
                    if not self._files:
                        self[exe.hash] = (self._compute_features(exe), True)  # True: force updating the row
                        exe.destination.remove(False)
                    i += 1
                    progress.update(pbar, advance=1)
                else:
                    del self[exe]
        ls = len(self)
        if ls > 0:
            p = sorted(list(set([lb for lb in self._data.label.values if isinstance(lb, str)])))
            l.info("Used packers: %s" % ", ".join(_ for _ in p if _ not in [NOT_LABELLED, NOT_PACKED]))
            l.info("#Executables: %d" % ls)
        if ls - n1 < n:
            l.warning("Found too few candidate executables")
        # finally, save dataset's metadata and labels to JSON files
        self._save()
    
    @backup
    def merge(self, name2=None, new_name=None, silent=False, **kw):
        """ Merge another dataset with the current one. """
        l = self.logger
        l_info = getattr(l, ["info", "debug"][silent])
        ds2 = Dataset.load(name2, name_check=False)
        if new_name is not None:
            dstype = FilelessDataset if not self._files or not ds2._files else Dataset
            ds = dstype(new_name)
            if dstype is FilelessDataset and self._files:
                temp_name = random.randstr(16)
                self.convert(temp_name, check=False)
                ds.merge(temp_name)
                FilelessDataset(temp_name, check=False)._purge()
            else:
                ds.merge(self.path.basename)
            if dstype is FilelessDataset and ds2._files:
                temp_name = random.randstr(16)
                ds2.convert(temp_name, check=False)
                ds.merge(temp_name)
                FilelessDataset(temp_name, check=False)._purge()
            else:
                ds.merge(name2)
            return
        if self.__class__ is Dataset and ds2.__class__ is FilelessDataset:
            l.error("Cannot merge a fileless dataset into a dataset (because files are missing)")
            return
        # add rows from the input dataset
        l_info("Merging rows from %s into %s..." % (ds2.basename, self.basename))
        with progress_bar(silent=silent) as p:
            for r in p.track(ds2):
                self[Executable(hash=r.hash, dataset=ds2, dataset2=self)] = r._row._asdict()
        # as the previous operation does not update formats and features, do it manually
        self._metadata.setdefault('formats', [])
        for fmt in ds2._metadata.get('formats', []):
            if fmt not in self._metadata['formats']:
                self._metadata['formats'].append(fmt)
        self._metadata['counts'] = self._data.label.value_counts().to_dict()
        self._metadata['executables'] = len(self)
        self._metadata.setdefault('sources', [])
        if str(ds2.path) not in self._metadata['sources']:
            self._metadata['sources'].append(str(ds2.path))
        if hasattr(self, "_features") and hasattr(ds2, "_features"):
            d = {k: v for k, v in ds2._features.items()}
            d.update(self._features)
            self._features = d
        self._save()
   
    def plot(self, subcommand=None, **kw):
        """ Plot something about the dataset. """
        # ensure input dataset(s) have their features computed before plotting
        if 'dataset' in kw and kw['dataset']._files:
            kw['dataset']._compute_all_features()
        if 'datasets' in kw:
            kw['datasets'] = [Dataset.load(ds) for ds in (kw['datasets'] or [])]
            [ds._compute_all_features() for ds in kw['datasets'] if ds._files]
        plot(self, "ds-%s" % subcommand, **kw)
    
    def preprocess(self, query=None, preprocessor=None, **kw):
        """ Preprocess dataset given selected features and preprocessors and display it with visidata for review. """
        self._compute_all_features()
        result = pd.DataFrame()
        for col in ["hash"] + Executable.FIELDS:
            result[col] = self._data[col]
        fnames = sorted(self._features.keys())
        data = self._data[fnames]
        if preprocessor:
            p = DebugPipeline()
            make_pipeline(p, preprocessor, self.logger)
            p.steps.append(("debug", DebugTransformer()))
            df = pd.DataFrame(p.preprocess(np.array(data)), columns=data.columns)
        for col in data.columns:
            if preprocessor:
                result[col] = df[col]
                col2 = "*" + col
                result[col2] = data[col]
            else:
                result[col] = data[col]
        result['label'] = self._data['label']
        with data_to_temp_file(filter_data(result, query, logger=self.logger), prefix="dataset-preproc-") as tmp:
            edit_file(tmp, logger=self.logger)
    
    @backup
    def remove(self, query=None, **kw):
        """ Remove executables from the dataset given multiple criteria. """
        self.logger.debug("removing files from %s based on query '%s'..." % (self.basename, query))
        for e in filter_data_iter(self._data, query, logger=self.logger):
            del self[e.hash]
        self._save()
    
    def rename(self, name2=None, **kw):
        """ Rename the current dataset. """
        l, path2 = self.logger, config['datasets'].joinpath(name2)
        if not path2.exists():
            l.debug("renaming %s (and backups) to %s..." % (self.basename, name2))
            tmp = TempPath(".dataset-backup", hex(hash(self))[2:])
            self.path.rename(path2)
            self.path = path2
            tmp2 = TempPath(".dataset-backup", hex(hash(self))[2:])
            tmp2.remove()
            tmp.rename(tmp2)
        else:
            l.warning("%s already exists" % name2)
    
    def revert(self, **kw):
        """ Revert to the latest version of the dataset (if a backup copy exists in /tmp). """
        b, l = self.backup, self.logger
        if b is None:
            l.warning("No backup found ; could not revert")
        else:
            l.debug("reverting %s to previous backup..." % self.basename)
            self._remove()
            b._copy(self.path)
            b._remove()
            self._save()
    
    def select(self, name2=None, query=None, limit=0, **kw):
        """ Select a subset from the current dataset based on multiple criteria. """
        self.logger.debug("selecting a subset of %s based on query '%s'..." % (self.basename, query))
        ds2 = self.__class__(name2)
        ds2._metadata['sources'] = []
        for e in filter_data_iter(self._data, query, limit, logger=self.logger):
            for s in self._metadata['sources']:
                if Path(e.realpath).is_under(Path(s, expand=True)) and s not in ds2._metadata['sources']:
                    ds2._metadata['sources'].append(s)
                    break
            ds2[Executable(dataset=self, dataset2=ds2, hash=e.hash)] = self[e.hash, True]
        ds2._save()
    
    def show(self, limit=10, per_format=False, **kw):
        """ Show an overview of the dataset. """
        self.__limit = limit if limit > 0 else len(self)
        self.__per_format = per_format
        if len(self) > 0:
            c = List(["**#Executables**: %d" % self._metadata['executables'],
                      "**Format(s)**:    %s" % ", ".join(self._metadata['formats']),
                      "**Packer(s)**:    %s" % (", ".join(get_counts(self._metadata).keys()) or "-"),
                      "**Size**:         %s" % human_readable_size(self.path.size),
                      "**Labelled**:     %.2f%%" % self.labelling])
            if len(self._alterations) > 0:
                c._data.append("**Altered**:      %d%%" % (int(round(100 * self._metadata['altered'], 0))))
                c._data.append("**Alterations**:  %s" % ", ".join(self._alterations.keys()))
            c._data.append("**With files**:   %s" % ["no", "yes"][self._files])
            r = [Section("Dataset characteristics"), c]
            r.extend(self.overview)
            render(*r)
        else:
            self.logger.warning("Empty dataset")
    
    @backup
    def update(self, source_dir=None, formats=["All"], n=0, labels=None, detect=False, **kw):
        """ Update the dataset with a folder of binaries, detecting used packers if 'detect' is set to True.
             If labels are provided, they are used instead of applying packer detection. """
        l, self.formats, labels = self.logger, formats, Dataset.labels_from_file(labels)
        if detect and not self._files:  # case of FilelessDataset ; cannot detect as files are not included
            l.warning("Label detection works only if the files are available")
            return
        if source_dir is None and len(labels) == 0:
            l.warning("No source folder provided")  # if labels are provided, source_dir can be None ; in this case,
            return                                  #  just apply provided labels to existing samples
        if not isinstance(source_dir, list):
            source_dir = [source_dir]
        def _update(e):
            # precedence: input dictionary of labels > dataset's own labels > detection (if enabled) > discard
            h = e.hash
            lbl = labels.get(h) or NOT_LABELLED
            # executable does not exist yet => create it without a label
            try:
                self._data[self._data.hash == h].iloc[0]
            except (AttributeError, IndexError):  # AttributeError occurs when no data yet (hence no headers)
                self[e] = NOT_LABELLED
            # label is found in the input labels dictionary and is not already NOT_LABELLED => update
            if lbl != NOT_LABELLED:
                self[e] = (lbl, True)
            # label was already set before => keep the former label
            elif e.label != NOT_LABELLED:
                pass
            # label was not found and is not set yet and detection is enabled => detect
            elif detect:
                self[e] = (Detector.detect(e), True)
            if not self._files:
                self[h] = (self._compute_features(e), True)  # True: force updating the row
        # case (1) source directories provided, eventually with labels => ingest samples
        #           labels available   => packed / not packed
        #           labels unavailable => not labelled
        if len(source_dir) > 0:
            silent = kw.get('silent', False)
            if not silent:
                l.info("Source directories: %s" % ",".join(map(str, set(source_dir))))
            self._metadata.setdefault('formats', [])
            self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + source_dir)))
            if n > 0:
                files = [p for p in self._walk(True, {'All': source_dir}, True)]
                random.shuffle(files)
                files, total = files[:n], n
            else:
                files = self._walk(True, {'All': source_dir}, silent)
                total = sum(1 for _ in self._walk(True, {'All': source_dir}, True))
            found = False
            with progress_bar() as p:
                for exe in p.track(files):
                    if exe.format not in self._metadata['formats']:
                        self._metadata['formats'].append(exe.format)
                    _update(exe)
                    found = True
            if not found:
                l.warning("No executable found")
        # case (2) empty source directory, but labels were provided and are to be applied
        else:
            with progress_bar() as p:
                for exe in p.track(self):
                    _update(exe)
        self._save()
    
    def view(self, query=None, **kw):
        """ View executables from the dataset given multiple criteria. """
        src = [Path(s, expand=True) for s in self._metadata.get('sources', [])]
        def _shorten(path):
            p = Path(path)
            for i, s in enumerate(src):
                if p.is_under(s):
                    return i, str(p.relative_to(s))
            return -1, path
        # prepare the table of records
        d, h = [], ["Hash", "Path", "Size", "Creation", "Modification", "Label"]
        for e in filter_data_iter(self._data, query, logger=self.logger):
            i, p = _shorten(e.realpath)
            if i >= 0:
                p = "[%d]/%s" % (i, p)
            d.append([e.hash, p, human_readable_size(e.size), e.ctime.strftime("%d/%m/%y"),
                      e.mtime.strftime("%d/%m/%y"), e.label])
        if len(d) == 0:
            return
        r = [Text("**Sources**:\n\n%s" % "\n".join("[%d] %s" % (i, s) for i, s in enumerate(src))),
             Section("Filtered records"), Table(d, column_headers=h)]
        render(*r)
    
    @property
    def backup(self):
        """ Get the latest backup. """
        l, bkp = self.logger, config['backup_copies']
        l.debug("backup %s" % ["disabled", "enabled (maximum copies: %d)" % bkp][bkp > 0])
        tmp = TempPath(".dataset-backup", hex(hash(self))[2:])
        l.debug("backup root: %s" % tmp)
        backups = sorted(tmp.listdir(self.__class__.check), key=lambda p: -int(p.basename))
        if len(backups) > 0:
            l.debug("> found: %s" % ", ".join(map(lambda p: p.basename, backups)))
        for backup in backups:
            return self.__class__(backup, check=False)
    
    @backup.setter
    def backup(self, dataset):
        """ Make a backup copy. """
        if len(self._data) == 0:
            return
        l = self.logger
        tmp = TempPath(".dataset-backup", hex(hash(self))[2:])
        l.debug("backup root: %s" % tmp)
        backups, i = [], 0
        for i, backup in enumerate(sorted(tmp.listdir(self.__class__.check), key=lambda p: -int(p.basename))):
            backup, n = self.__class__(backup, check=False), 0
            # if there is a change since the last backup, create a new one
            if i == 0 and dataset != backup:
                name = int(time.time())
                l.debug("> creating backup %d" % name)
                dataset._copy(tmp.joinpath(str(name)))
                n = 1
            elif i >= config['backup_copies'] - n:
                l.debug("> removing backup %s (limit: %d)" % (backup.basename, config['backup_copies']))
                backup._remove()
        if i == 0:
            name = int(time.time())
            l.debug("> creating backup %d" % name)
            dataset._copy(tmp.joinpath(str(name)))
    
    @property
    def files(self):
        """ Get the Path instance for the 'files' folder. """
        if self._files or self.__class__ is Dataset:
            return self.path.joinpath("files")
        raise AttributeError("'FilelessDataset' object has no attribute 'files'")
    
    @property
    def labelling(self):
        """ Get the percentage of labels set. """
        return 100 * sum(get_counts(self._metadata, False).values()) / self._metadata['executables']
    
    @property
    def labels(self):
        """ Get the series of labels. """
        return self._data.label
    
    @property
    def name(self):
        """ Get the name of the dataset composed with its list of formats. """
        fmt, p = getattr(self, "formats", []), self.path.basename
        return "%s(%s)" % (p, ",".join(sorted(collapse_formats(*fmt)))) if len(fmt) > 0 else self.path.basename
    
    @property
    def overview(self):
        """ Represent an overview of the dataset. """
        self.logger.debug("computing dataset overview...")
        r = []
        CAT = ["<20kB", "20-50kB", "50-100kB", "100-500kB", "500kB-1MB", ">1MB"]
        size_cat = lambda s: CAT[0] if s < 20 * 1024 else CAT[1] if 20 * 1024 <= s < 50 * 1024 else \
                             CAT[2] if 50 * 1024 <= s < 100 * 1024 else CAT[3] if 100 * 1024 <= s < 500 * 1024 else \
                             CAT[4] if 500 * 1024 <= s < 1024 * 1024 else CAT[5]
        # data1: per size range, statistics for not labelled, not packed and packed binaries
        # data2: per binary, hash|path|ctime|mtime|label
        data1, data2 = {}, {}
        formats = expand_formats("All") if self.__per_format else ["All"]
        src = self._metadata.get('sources', [])
        r = []
        def _shorten(path):
            p = Path(path)
            for i, s in enumerate(src):
                if p.is_under(s):
                    return i, str(p.relative_to(s))
            return -1, path
        # parse formats, collect counts per size range and list of files
        #  counts list: not labelled, not packed, packed
        Executable._metadata_only = True
        for fmt in formats:
            d = {c: [0, 0, 0] for c in CAT}
            for e in self:
                if fmt != "All" and e.format != fmt:
                    continue
                l, s = "not packed" if str(e.label) == NOT_PACKED else e.label, size_cat(e.size)
                # not labelled
                if e.label == NOT_LABELLED:
                    d[s][0] += 1
                    l = "?"
                # not packed
                elif e.label == NOT_PACKED:
                    d[s][1] += 1
                    l = "-"
                # packed
                else:
                    d[s][2] += 1
                    l = e.label
                # binary-related information
                data2.setdefault(fmt, [])
                if len(data2[fmt]) < self.__limit:
                    i, p = _shorten(e.realpath)
                    if i >= 0:
                        p = "[%d]/%s" % (i, p)
                    row = [e.hash, p, e.ctime.strftime("%d/%m/%y"), e.mtime.strftime("%d/%m/%y"), l]
                    data2[fmt].append(row)
                elif len(data2[fmt]) == self.__limit:
                    data2[fmt].append(["...", "...", "...", "...", "..."])
            totalnl, totalnp, totalp = (sum([v[x] for v in d.values()]) for x in range(3))
            total = totalnl + totalnp + totalp
            if total == 0:
                continue
            # per-size-range statitics
            data1.setdefault(fmt, [])
            percent = lambda x, y: "%.2f" % (100 * (float(x) / y)) if y > 0 else 0
            for c in CAT:
                # 0:   category
                # 1,2: count and percentage of not labelled
                # 3,4: count and percentage of not packed
                # 5,6: count and percentage of packed
                data1[fmt].append([c, d[c][0], percent(d[c][0], totalnl), d[c][1], percent(d[c][1], totalnp),
                                      d[c][2], percent(d[c][2], totalp)])
            data1[fmt].append(["Total", str(totalnl), percent(totalnl, total), str(totalnp), percent(totalnp, total),
                               str(totalp), percent(totalp, total)])
        # display statistics if any (meaning that null stats are filtered out)
        if len(data1) > 0:
            r.append(Section("Executables per size%s" % ["" ," and format"][self.__per_format]))
            for fmt in formats:
                fmt = fmt if self.__per_format else "All"
                if fmt in data1:
                    if fmt != "All":
                        r.append(Subsection(fmt))
                    h = ["Size\nRange", "Not\nLabelled", "%", "Not\nPacked", "%", "Packed", "%"]
                    if data1[fmt][-1][3] == "0" and data1[fmt][-1][5] == "0":  # all not labelled
                        h, d = h[:3], [[r[0], r[1], r[2]] for r in data1[fmt]]
                    elif data1[fmt][-1][1] == "0" and data1[fmt][-1][5] == "0":  # all not packed
                        h, d = [h[0]] + h[3:5], [[r[0], r[3], r[4]] for r in data1[fmt]]
                    elif data1[fmt][-1][1] == "0" and data1[fmt][-1][3] == "0":  # all packed
                        h, d = [h[0]] + h[5:7], [[r[0], r[5], r[6]] for r in data1[fmt]]
                    elif data1[fmt][-1][5] == "0":  # none packed
                        h, d = h[:6], [r[:5] for r in data1[fmt]]
                    elif data1[fmt][-1][3] == "0":  # none not packed
                        h, d = h[:3] + h[5:7], [[r[0], r[1], r[2], r[5], r[6]] for r in data1[fmt]]
                    elif data1[fmt][-1][1] == "0":  # none not labelled
                        h, d = [h[0]] + h[3:7], [[r[0], r[3], r[4], r[5], r[6]] for r in data1[fmt]]
                    r += [Table(d, column_headers=h)]
                if fmt == "All":
                    break
        r.append(Rule())
        r.append(Text("**Sources**:\n\n%s" % "\n".join("[%d] %s" % (i, s) for i, s in enumerate(src))))
        # display files if any
        if len(data2) > 0:
            r.append(Section("Executables' metadata and labels"))
            for fmt in formats:
                fmt = fmt if self.__per_format else "All"
                if fmt in data2:
                    if fmt != "All":
                        r.append(Subsection(fmt))
                    r += [Table(data2[fmt], column_headers=["Hash", "Path", "Creation", "Modification", "Label"])]
                if fmt == "All":
                    break
        return r
    
    @classmethod
    def list(cls, show_all=False, hide_files=False, raw=False, **kw):
        """ List all the datasets from the given path. """
        if raw:
            for name in sorted([dset.basename] for dset in Path(config['datasets']) \
                                .listdir(lambda p: Dataset.check(p) or FilelessDataset.check(p))):
                print(name)
        else:
            l = cls.logger
            l.debug("summarizing datasets from %s..." % config['datasets'])
            section, table = FilelessDataset.summarize(show_all, hide_files)
            if section is not None and table is not None:
                render(section, table)
            else:
                l.warning("No dataset found in the workspace (%s)" % config['datasets'])
    
    @classmethod
    def validate(cls, folder, **kw):
        f = Path(folder, expand=True)
        if not f.is_dir():
            f = config['datasets'].joinpath(folder)
            if not f.exists():
                raise ValueError("Folder does not exist")
            if not f.is_dir():
                raise ValueError("Input is not a folder")
        if not f.joinpath("files").is_dir():
            raise ValueError("Missing 'files' folder")
        for fn in ["data.csv", "metadata.json"]:
            if not f.joinpath(fn).exists():
                raise ValueError("'%s' does not exist" % fn)
        return f
    
    @staticmethod
    def labels_from_file(labels):
        labels = labels or {}
        if isinstance(labels, str):
            labels = Path(labels)
        if isinstance(labels, Path) and labels.is_file():
            with labels.open() as f:
                labels = json.load(f)
        if not isinstance(labels, dict):
            raise ValueError("Bad labels ; not a dictionary or JSON file")
        return {h: l or NOT_PACKED for h, l in labels.items()}
    
    #TODO: refactor (dates from pbox structure with 'common' and 'learning' subpackages, each containing respectively
    #       Dataset and FilelessDataset)
    @staticmethod
    def summarize(show=False, hide_files=False, check_func=None):
        datasets, headers = [], ["Name", "#Executables", "Size"] + [["Files"], []][hide_files] + ["Formats", "Packers"]
        for dset in Path(config['datasets']).listdir(check_func or Dataset.check):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                row = [
                    dset.basename,
                    str(metadata['executables']),
                    human_readable_size(dset.size),
                ] + [[["no", "yes"][dset.joinpath("files").exists()]], []][hide_files] + [
                    ",".join(sorted(metadata['formats'])),
                    shorten_str(",".join("%s{%d}" % i for i in sorted(get_counts(metadata).items(),
                                                                      key=lambda x: (-x[1], x[0])))),
                ]
            except Exception as err:
                row = None
                if show:
                    if headers[-1] != "Reason":
                        headers.append("Reason")
                    row = [
                        dset.basename,
                        str(metadata.get('executables', colored("?", "red"))),
                        human_readable_size(dset.size),
                    ] + [[["no", "yes"][dset.joinpath("files").exists()]], []][hide_files] + [
                        colored("?", "red"), colored("?", "red"),
                        colored("%s: %s" % (err.__class__.__name__, str(err)), "red")
                    ]
            if row:
                datasets.append(row)
        n = len(datasets)
        if n > 0:
            return [Section("Datasets (%d)" % n), Table(datasets, column_headers=headers)]
        return None, None


class FilelessDataset(Dataset):
    """ Folder structure:
    
    [name]
      +-- data.csv          # metadata and labels of the executable
      +-- features.json     # dictionary of selected features and their descriptions
      +-- metadata.json     # simple statistics about the dataset
    """
    _files = False
    
    @classmethod
    def validate(cls, folder, **kw):
        f = Path(folder, expand=True)
        if not f.is_dir():
            f = config['datasets'].joinpath(folder)
            if not f.exists():
                raise ValueError("Folder does not exist")
            if not f.is_dir():
                raise ValueError("Input is not a folder")
        if f.joinpath("files").is_dir():
            raise ValueError("Has 'files' folder while it should not")
        for fn in ["data.csv", "metadata.json", "features.json"]:
            if not f.joinpath(fn).exists():
                raise ValueError("'%s' does not exist" % fn)
        return f
    
    #TODO: refactor (dates from pbox structure with 'common' and 'learning' subpackages, each containing respectively
    #       Dataset and FilelessDataset)
    @staticmethod
    def summarize(show=False, hide_files=False):
        _, table = Dataset.summarize(show, hide_files)
        _, table2 = Dataset.summarize(show, hide_files, FilelessDataset.check)
        t, t2 = [] if table is None else table.data, [] if table2 is None else table2.data
        datasets = sorted(t + t2, key=lambda x: x[0])
        if len(datasets) > 0:
            table = Table(datasets, column_headers=(table or table2).column_headers)
            return [Section("Datasets (%d)" % len(table.data)), table]
        return None, None

