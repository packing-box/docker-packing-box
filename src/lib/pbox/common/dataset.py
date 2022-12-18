# -*- coding: UTF-8 -*-
import pandas as pd
import re
from datetime import datetime, timedelta
from textwrap import wrap
from tinyscript import b, colored, hashlib, json, logging, random, subprocess, time, ts
from tinyscript.report import *
from tqdm import tqdm

from .config import *
from .executable import *
from .modifiers import *
from .utils import *
from ..items import *


__all__ = ["Dataset"]


BACKUP_COPIES = 3


class Dataset:
    """ Folder structure:
    
    [name]
      +-- files
      |     +-- {executables, renamed to their SHA256 hashes}
      +-- data.csv            # metadata and labels of the executable
      +-- metadata.json       # simple statistics about the dataset
      +-- (alterations.json)  # if the dataset was altered, this contains the hashes of the altered executables
    """
    @logging.bindLogger
    def __init__(self, name="dataset", source_dir=None, load=True, **kw):
        if not re.match(NAMING_CONVENTION, name.basename if isinstance(name, ts.Path) else str(name)):
            raise ValueError("Bad input name (%s)" % name)
        self._files = getattr(self.__class__, "_files", True)
        self.path = ts.Path(config['datasets'].joinpath(name), create=load).absolute()
        self.sources = source_dir or PACKING_BOX_SOURCES
        if isinstance(self.sources, list):
            self.sources = {'All': [str(x) for x in self.sources]}
        for _, sources in self.sources.items():
            for source in sources[:]:
                s = ts.Path(source, expand=True)
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
                ext = ts.Path(df.loc[df['hash'] == h, 'realpath'].iloc[0]).extension
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
        return int.from_bytes(hashlib.md5(b(self.name)).digest(), "little")
    
    def __len__(self):
        """ Get dataset's length. """
        return len(self._data)
    
    def __repr__(self):
        """ Custom string representation. """
        return "<%s dataset at 0x%x>" % (self.name, id(self))
    
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
                for n, v in d.items():
                    df.loc[df['hash'] == e.hash, n] = v
            else:
                l.debug("discarding %s%s..." % (e.hash, ["", " (already in dataset)"][lbl == d['label']]))
        else:
            l.debug("adding %s..." % e.hash)
            self._data = df.append(d, ignore_index=True)
    
    def __str__(self):
        """ Custom object's string. """
        return self.name
    
    def _copy(self, path):
        """ Copy the current dataset to a given destination. """
        self.path.copy(path)
    
    def _filter(self, query=None, **kw):
        """ Yield executables' hashes from the dataset using Pandas' query language. """
        i = -1
        try:
            for i, row in enumerate(self._data.query(query or "()").itertuples()):
                yield row
            if i == -1:
                self.logger.warning("No data selected")
        except (AttributeError, KeyError) as e:
            self.logger.error("Invalid query syntax ; %s" % e)
        except SyntaxError:
            self.logger.error("Invalid query syntax ; please checkout Pandas' documentation for more information")
        except pd.errors.UndefinedVariableError as e:
            self.logger.error(e)
            self.logger.info("Possible values:\n%s" % "".join("- %s\n" % n for n in self._data.columns))
    
    def _load(self):
        """ Load dataset's associated files or create them. """
        try:
            if len(self) > 0:
                self.logger.debug("loading dataset '%s'..." % self.basename)
        except AttributeError:  # self._data does not exist yet
            pass
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
    
    def _remove(self):
        """ Remove the current dataset. """
        self.logger.debug("removing dataset '%s'..." % self.basename)
        self.path.remove(error=False)
    
    def _save(self):
        """ Save dataset's state to JSON files. """
        if not self.__change:
            return
        if len(self) == 0 and not Dataset.check(self.basename):
            self._remove()
            return
        self.logger.debug("saving dataset '%s'..." % self.basename)
        self._metadata['formats'] = sorted(collapse_formats(*self._metadata['formats']))
        try:
            self._metadata['counts'] = self._data.label.value_counts().to_dict()
        except AttributeError:
            self.logger.warning("No label found")
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
        [self.logger.info, self.logger.debug][silent]("Searching for executables...")
        m, candidates, packers = 0, [], [p.name for p in Packer.registry]
        for cat, srcs in (sources or self.sources).items():
            if all(c not in expand_formats(cat) for c in self._formats_exp):
                continue
            for src in srcs:
                for exe in ts.Path(src, expand=True).walk(filter_func=lambda x: x.is_file(), sort=False):
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
    def alter(self, new_name=None, percentage=.1, **kw):
        """ Alter executables with some given modifiers. """
        l = self.logger
        if not self._files:
            l.warning("Modifiers only work on a normal dataset (not on a fileless one)")
            return
        if new_name is not None:
            ds = Dataset(new_name)
            ds.merge(self.path.basename, silent=True, **kw)
            ds.alter(**kw)
            return
        # keep previous alteration percentage into account
        a = self._metadata.get('altered', .0)
        p = min(1. - a, percentage)
        p_ = round(p * 100, 0)
        if p != percentage:
            if p == .0:
                l.warning("Nothing more to alter")
                return
            else:
                l.warning("Setting alterations percentage to %d" % p_)
        l.info("Altering %d%% of the dataset..." % p_)
        hashes = self._data.hash.values[:]
        # randomly sort hashes
        if p < 1.:
            random.shuffle(hashes)
        # then apply alterations until the desired percentage is reached
        n, c = int(round(len(self)*p, 0)), 0
        for h in hashes:
            if any(h in altered_hs for altered_hs in self._alterations.values()):
                continue
            exe = Executable(dataset=self, hash=h)
            for m in Modifiers(exe):
                self._alterations.setdefault(m, [])
                self._alterations[m].append(h)
            c += 1
            if c >= n:
                break
        self._metadata['altered'] = a + c / len(self)
        self._save()
    
    def edit(self, **kw):
        """ Edit the data CSV file. """
        self.logger.debug("editing dataset's data.csv...")
        edit_file(self.path.joinpath("data.csv").absolute(), logger=self.logger)
    
    def exists(self):
        """ Dummy exists method. """
        return self.path.exists()
    
    def export(self, destination="export", n=0, **kw):
        """ Export packed executables to the given destination folder. """
        self.logger.info("Exporting packed executables of %s to '%s'..." % (self.basename, destination))
        l, tmp = [e for e in self if e.label not in [NOT_PACKED, NOT_LABELLED]], []
        n = min(n, len(l))
        if n > 0:
            random.shuffle(l)
        pbar = tqdm(total=n or len(l), unit="packed executable")
        for i, exe in enumerate(l):
            if i >= n > 0:
                break
            fn = "%s_%s" % (exe.label, ts.Path(exe.realpath).filename)
            if fn in tmp:
                self.logger.warning("duplicate '%s'" % fn)
                n += 1
                continue
            exe.destination.copy(ts.Path(destination, create=True).joinpath(fn))
            tmp.append(fn)
            pbar.update()
        pbar.close()
    
    @backup
    def fix(self, **kw):
        """ Make dataset's structure and files match. """
        self.logger.debug("dropping duplicates...")
        self._data = self._data.drop_duplicates()
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
            elif not self.files.joinpath(h).exists():
                del self[h]
        self._save()
    
    def is_empty(self):
        """ Check if this Dataset instance has a valid structure. """
        return len(self) == 0
    
    def is_valid(self):
        """ Check if this Dataset instance has a valid structure. """
        return self.__class__.check(self.path)
    
    def list(self, show_all=False, hide_files=False, raw=False, **kw):
        """ List all the datasets from the given path. """
        self.logger.debug("summarizing datasets from %s..." % config['datasets'])
        d = Dataset.summarize(str(config['datasets']), show_all, hide_files)
        if len(d) > 0:
            r = mdv.main(Report(*d).md())
            print(ts.ansi_seq_strip(r) if raw else r)
        else:
            self.logger.warning("No dataset found in workspace (%s)" % config['datasets'])
    
    @backup
    def make(self, n=0, formats=["All"], balance=False, packer=None, pack_all=False, **kw):
        """ Make n new samples in the current dataset among the given binary formats, balanced or not according to
             the number of distinct packers. """
        l, self.formats = self.logger, formats  # this triggers creating self._formats_exp
        # select enabled and non-failing packers among the input list
        packers = [p for p in (packer or Packer.registry) if p in Packer.registry and \
                                                             p.check(*self._formats_exp, silent=False)]
        if len(packers) == 0:
            l.critical("No valid packer selected")
            return
        # then restrict dataset's formats to these of the selected packers
        pformats = aggregate_formats(*[p.formats for p in packers])
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
        i, cbad, cgood, pbar = 0, {p: CBAD for p in packers}, {p: CGOOD for p in packers}, None
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
            if len(self._data) > 0 and exe.destination.exists():
                l.debug("already in the dataset (%s)" % exe)
                continue
            l.debug("handling %s..." % exe)
            # set the progress bar now to not overlap with self._walk's logging
            if pbar is None:
                pbar = tqdm(total=n, unit="executable")
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
                i += 1
                pbar.update()
            else:
                del self[exe]
        if pbar:
            pbar.close()
        ls = len(self)
        if ls > 0:
            p = sorted(list(set([lb for lb in self._data.label.values if isinstance(lb, str)])))
            l.info("Used packers: %s" % ", ".join(_ for _ in p if _ != NOT_PACKED))
            l.info("#Executables: %d" % ls)
        if ls - n1 < n:
            l.warning("Found too few candidate executables")
        # finally, save dataset's metadata and labels to JSON files
        self._save()
    
    def purge(self, backup=False, **kw):
        """ Truncate and recreate a blank dataset. """
        self.logger.debug("purging %s%s..." % (self.path, ["", "'s backups"][backup]))
        if not backup:
            self._remove()
        # also recursively purge the backups
        try:
            self.backup.purge()
        except AttributeError:
            pass
    
    @backup
    def remove(self, query=None, **kw):
        """ Remove executables from the dataset given multiple criteria. """
        self.logger.debug("removing files from %s based on query '%s'..." % (self.basename, query))
        for e in self._filter(query, **kw):
            del self[e.hash]
        self._save()
    
    def rename(self, name2=None, **kw):
        """ Rename the current dataset. """
        l, path2 = self.logger, config['datasets'].joinpath(name2)
        if not path2.exists():
            l.debug("renaming %s (and backups) to %s..." % (self.basename, name2))
            tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
            self.path.rename(path2)
            self.path = path2
            tmp.rename(ts.TempPath(".dataset-backup", hex(hash(self))[2:]))
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
        ds2._metadata['sources'] = self._metadata['sources'][:]
        _tmp, i = {s: 0 for s in ds2._metadata['sources']}, 0
        for e in self._filter(query, **kw):
            if i >= limit > 0:
                break
            for s in ds2._metadata['sources']:
                if e.realpath.startswith(s):
                    _tmp[s] += 1
                    break
            ds2[Executable(dataset=self, dataset2=ds2, hash=e.hash)] = self[e.hash, True]
            i += 1
        # remove sources that, given the selection, have no associated file
        for s, cnt in _tmp.items():
            if cnt == 0:
                ds2._metadata['sources'].remove(s)
        ds2._save()
    
    def show(self, limit=10, per_format=False, **kw):
        """ Show an overview of the dataset. """
        self.__limit = limit if limit > 0 else len(self)
        self.__per_format = per_format
        if len(self) > 0:
            c = List(["**#Executables**: %d" % self._metadata['executables'],
                      "**Format(s)**:    %s" % ", ".join(self._metadata['formats']),
                      "**Packer(s)**:    %s" % ", ".join(x for x in self._metadata['counts'].keys() if x != NOT_PACKED),
                      "**Size**:         %s" % ts.human_readable_size(self.path.size),
                      "**Labelled**:     %.2f%%" % self.labelling])
            if len(self._alterations) > 0:
                c._data.append("**Altered**:      %d%%" % (int(round(100 * self._metadata['altered'], 0))))
                c._data.append("**Alterations**:  %s" % ", ".join(self._alterations.keys()))
            c._data.append("**With files**:   %s" % ["no", "yes"][self._files])
            r = Report(Section("Dataset characteristics"), c)
            r.extend(self.overview)
            print(mdv.main(r.md()))
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
            if h not in self:
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
        # case (1) source directories provided, eventually with labels => ingest samples
        #           labels available   => packed / not packed
        #           labels unavailable => not labelled
        if len(source_dir) > 0:
            l.info("Source directories: %s" % ",".join(map(str, set(source_dir))))
            self._metadata.setdefault('formats', [])
            self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + source_dir)))
            i, pbar = -1, None
            if n > 0:
                files = [p for p in self._walk(True, {'All': source_dir}, True)]
                random.shuffle(files)
                files, total = files[:n], n
            else:
                files = self._walk(True, {'All': source_dir})
                total = sum(1 for _ in self._walk(True, {'All': source_dir}, True))
            for i, exe in enumerate(files):
                # set the progress bar now to not overlap with self._walk's logging
                if pbar is None:
                    pbar = tqdm(total=total, unit="executable")
                if exe.format not in self._metadata['formats']:
                    self._metadata['formats'].append(exe.format)
                _update(exe)
                pbar.update()
            if pbar:
                pbar.close()
            if i < 0:
                l.warning("No executable found")
        # case (2) empty source directory, but labels were provided and are to be applied
        else:
            pbar = None
            for exe in self:
                if pbar is None:
                    pbar = tqdm(total=len(self), unit="executable")
                _update(exe)
                pbar.update()
            if pbar:
                pbar.close()
        self._save()
    
    def view(self, query=None, **kw):
        """ View executables from the dataset given multiple criteria. """
        src = self._metadata.get('sources', [])
        def _shorten(path):
            p = ts.Path(path)
            for i, s in enumerate(src):
                if p.is_under(s):
                    return i, str(p.relative_to(s))
            return -1, path
        # prepare the table of records
        d, h = [], ["Hash", "Path", "Size", "Creation", "Modification", "Label"]
        Executable._metadata_only = True
        for e in self._filter(query, **kw):
            e = Executable(dataset=self, hash=e.hash)
            i, p = _shorten(e.realpath)
            if i >= 0:
                p = "[%d]/%s" % (i, p)
            d.append([e.hash, p, ts.human_readable_size(e.size), e.ctime.strftime("%d/%m/%y"),
                      e.mtime.strftime("%d/%m/%y"), e.label])
        if len(d) == 0:
            return
        r = [Text("Sources:\n%s" % "\n".join("[%d] %s" % (i, s) for i, s in enumerate(src))),
             Table(d, title="Filtered records", column_headers=h)]
        print(mdv.main(Report(*r).md()))
    
    @property
    def backup(self):
        """ Get the latest backup. """
        tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
        for backup in sorted(tmp.listdir(self.__class__.check), key=lambda p: -int(p.basename)):
            return self.__class__(backup)
    
    @backup.setter
    def backup(self, dataset):
        """ Make a backup copy. """
        if len(self._data) == 0:
            return
        tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
        backups, i = [], 0
        for i, backup in enumerate(sorted(tmp.listdir(Dataset.check), key=lambda p: -int(p.basename))):
            backup, n = self.__class__(backup), 0
            # if there is a change since the last backup, create a new one
            if i == 0 and dataset != backup:
                dataset._copy(tmp.joinpath(str(int(time.time()))))
                n = 1
            elif i >= BACKUP_COPIES - n:
                backup._remove()
        if i == 0:
            dataset._copy(tmp.joinpath(str(int(time.time()))))
    
    @property
    def files(self):
        """ Get the Path instance for the 'files' folder. """
        if self._files or self.__class__ is Dataset:
            return self.path.joinpath("files")
        raise AttributeError("'FilelessDataset' object has no attribute 'files'")
    
    @property
    def labelling(self):
        """ Get the percentage of labels set. """
        return 100 * sum(self._metadata['counts'].values()) / self._metadata['executables']
    
    @property
    def labels(self):
        """ Get the series of labels. """
        return self._data.label
    
    @property
    def basename(self):
        """ Dummy shortcut for dataset's path.basename. """
        return self.path.basename
    
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
            p = ts.Path(path)
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
            for c in CAT:
                # 0:   category
                # 1,2: count and percentage of not labelled
                # 3,4: count and percentage of not packed
                # 5,6: count and percentage of packed
                data1[fmt].append([c, d[c][0], "%.2f" % (100 * (float(d[c][0]) / totalnl)) if totalnl > 0 else 0,
                                      d[c][1], "%.2f" % (100 * (float(d[c][1]) / totalnp)) if totalnp > 0 else 0,
                                      d[c][2], "%.2f" % (100 * (float(d[c][2]) / totalp)) if totalp > 0 else 0])
            data1[fmt].append(["Total", str(totalnl), "", str(totalnp), "", str(totalp), ""])
        # display statistics if any (meaning that null stats are filtered out)
        if len(data1) > 0:
            r.append(Section("Executables per size%s" % ["" ," and format"][self.__per_format]))
            for fmt in formats:
                fmt = fmt if self.__per_format else "All"
                if fmt in data1:
                    if fmt != "All":
                        r.append(Subsection(fmt))
                    # all not labelled (so not_packed=0 and packed=0)
                    if data1[fmt][-1][3] == "0" and data1[fmt][-1][5] == "0":
                        h = ["Size Range", "Not Labelled", "%"]
                        d = [[r[0], r[1], r[2]] for r in data1[fmt]]
                    # all not packed (so not_labelled=0 and packed=0)
                    elif data1[fmt][-1][1] == "0" and data1[fmt][-1][5] == "0":
                        h = ["Size Range", "Not Packed", "%"]
                        d = [[r[0], r[3], r[4]] for r in data1[fmt]]
                    # all packed (so not_labelled=0 and not_packed=0)
                    elif data1[fmt][-1][1] == "0" and data1[fmt][-1][3] == "0":
                        h = ["Size Range", "Packed", "%"]
                        d = [[r[0], r[5], r[6]] for r in data1[fmt]]
                    # none packed
                    elif data1[fmt][-1][5] == "0":
                        h = ["Size Range", "Not Labelled", "%", "Not Packed", "%"]
                        d = [r[:5] for r in data1[fmt]]
                    # none not packed
                    elif data1[fmt][-1][3] == "0":
                        h = ["Size Range", "Not Labelled", "%", "Packed", "%"]
                        d = [[r[0], r[1], r[2], r[5], r[6]] for r in data1[fmt]]
                    # none not labelled
                    elif data1[fmt][-1][1] == "0":
                        h = ["Size Range", "Not Packed", "%", "Packed", "%"]
                        d = [[r[0], r[3], r[4], r[5], r[6]] for r in data1[fmt]]
                    # mix of all
                    else:
                        h = ["Size Range", "Not Labelled", "%", "Not Packed", "%", "Packed", "%"]
                    r += [Table(d, title=fmt, column_headers=h)]
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
                    r += [Table(data2[fmt], title=fmt,
                                column_headers=["Hash", "Path", "Creation", "Modification", "Label"])]
                if fmt == "All":
                    break
        return r
    
    @classmethod
    def check(cls, name):
        try:
            cls.validate(name, False)
            return True
        except ValueError as e:
            return False
    
    @classmethod
    def iteritems(cls):
        s = cls.summarize(str(config['datasets']), False)
        if len(s) > 0:
            for row in s[1].data:
                yield Dataset(row[0])
    
    @classmethod
    def validate(cls, name, load=True):
        f = getattr(cls, "_files", True)
        ds = cls(name, load=False)
        p = ds.path
        if not p.is_dir():
            raise ValueError
        if f and not p.joinpath("files").is_dir() or not f and p.joinpath("files").is_dir():
            raise ValueError
        for fn in ["data.csv", "metadata.json"] + [["features.json"], []][f]:
            if not p.joinpath(fn).exists():
                raise ValueError
        if load:
            ds._load()
        return ds
    
    @staticmethod
    def labels_from_file(labels):
        labels = labels or {}
        if isinstance(labels, str):
            labels = ts.Path(labels)
        if isinstance(labels, ts.Path) and labels.is_file():
            with labels.open() as f:
                labels = json.load(f)
        if not isinstance(labels, dict):
            raise ValueError("Bad labels ; not a dictionary or JSON file")
        return {h: l or NOT_PACKED for h, l in labels.items()}
    
    @staticmethod
    def summarize(path=None, show=False, hide_files=False):
        datasets, headers = [], ["Name", "#Executables", "Size"] + [["Files"], []][hide_files] + ["Formats", "Packers"]
        for dset in ts.Path(config['datasets']).listdir(lambda x: x.joinpath("metadata.json").exists()):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                counts = {k: v for k, v in metadata['counts'].items() if k != NOT_PACKED}
                row = [
                    dset.basename,
                    str(metadata['executables']),
                    ts.human_readable_size(dset.size),
                ] + [[["no", "yes"][dset.joinpath("files").exists()]], []][hide_files] + [
                    ",".join(sorted(metadata['formats'])),
                    shorten_str(",".join("%s{%d}" % i for i in sorted(counts.items(), key=lambda x: (-x[1], x[0])))),
                ]
            except Exception as err:
                row = None
                if show:
                    if headers[-1] != "Reason":
                        headers.append("Reason")
                    row = [
                        dset.basename,
                        str(metadata.get('executables', colored("?", "red"))),
                        ts.human_readable_size(dset.size),
                    ] + [[["no", "yes"][dset.joinpath("files").exists()]], []][hide_files] + [
                        colored("?", "red"), colored("?", "red"),
                        colored("%s: %s" % (err.__class__.__name__, str(err)), "red")
                    ]
            if row:
                datasets.append(row)
        n = len(datasets)
        return [] if n == 0 else [Section("Datasets (%d)" % n), Table(datasets, column_headers=headers)]

