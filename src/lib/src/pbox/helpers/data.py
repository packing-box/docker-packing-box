# -*- coding: UTF-8 -*-
from tinyscript import functools, json, re
from tinyscript.helpers import is_function

from .utils import np, pd


__all__ = ["filter_data", "filter_data_iter", "filter_features", "get_data", "make_test_dataset", "pd", "reduce_data"]


def filter_data(df, query=None, feature=None, **kw):
    """ Fitler an input Pandas DataFrame based on a given query. """
    r, l = None, kw.get('logger', null_logger)
    try:
        r = df if query is None or query.lower() == "all" else df.query(query)
        if len(r) == 0 and not kw.get('silent', False):
            l.warning("No data selected")
    except (AttributeError, KeyError) as e:
        l.error(f"Invalid query syntax ; {e}")
    except SyntaxError:
        l.error("Invalid query syntax ; please checkout Pandas' documentation for more information")
    except pd.errors.UndefinedVariableError as e:
        l.error(e)
        l.info("Possible values:\n%s" % "".join("- %s\n" % n for n in df.columns))
    if r is not None:
        if feature:
            r = r[["hash"] + EXE_METADATA + feature + ["label"]]
        return r


def filter_data_iter(df, query=None, feature=None, limit=0, sample=True, progress=True, target=None, transient=False,
                     **kw):
    """ Generator for the filtered data from an input Pandas DataFrame based on a given query. """
    from .rendering import progress_bar
    df = filter_data(df, query, feature, **kw)
    if df is None:
        return
    n = len(df.index)
    limit = n if limit <= 0 else min(n, limit)
    if sample and limit < n:
        df = df.sample(n=limit)
    i, generator = 0, filter_data(df, query, feature, silent=True, **kw).itertuples()
    if progress:
        with progress_bar(target=target, transient=transient) as p:
            task = p.add_task("", total=limit)
            for row in generator:
                yield row
                i += 1
                p.update(task, advance=1.)
                if i >= limit > 0:
                    break
    else:
        for row in generator:
            yield row
            i += 1
            if i >= limit > 0:
                break


def filter_features(dataset, feature=None):
    """ Handle features selection based on a simple wildcard. """
    if not hasattr(dataset, "_features"):
        dataset._compute_all_features()
    if feature is None or feature == []:
        feature = list(dataset._features.keys())
    # data preparation
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    nfeature = []
    for pattern in feature:
        # handle wildcard for bulk-selecting features
        if "*" in pattern:
            regex = re.compile(pattern.replace("*", ".*"))
            for f in dataset._features.keys():
                if regex.search(f):
                    nfeature.append(f)
        else:
            nfeature.append(pattern)
    return sorted(nfeature)


@functools.lru_cache
def get_data(exe_format):
    """ Prepare data for a particular executable format.
    
    Convention for subfolder and file naming: lower() and remove {"-", "_", "."}
      PE     => pe
      Mach-O => macho
    
    Examples:
      Let us take the following structure (NB: ~/.packing-box/data is the default location defined in config):
        data/
         +-- pe/
         |    +-- common_dll_imports.txt
         |    +-- common_dll_imports_pe64.txt
         |    +-- common_section_names.txt
         +-- common_section_names.txt
      
     get_data("PE") will output:
      {
        'COMMON_DLL_IMPORTS':   <list from data/pe/common_dll_imports.txt>
        'COMMON_SECTION_NAMES': <list from data/pe/common_section_names.txt>
      }
     
     get_data("ELF") will output:
      {
        'COMMON_SECTION_NAMES': <list from data/common_section_names.txt>
      }
      
     get_data("PE64") will output:
      {
        'COMMON_DLL_IMPORTS':   <list from data/pe/common_dll_imports_pe64.txt>
        'COMMON_SECTION_NAMES': <list from data/pe/common_section_names.txt>
      }
    """
    from .formats import format_shortname as _name, get_format_group
    _add = lambda a, b: _sort(a.update(b)) if isinstance(a, dict) else list(set(a + b))
    _sort = lambda d: dict(sorted(d.items())) if isinstance(d, dict) else sorted(d or {})
    _uncmt = lambda s: s.split("#", 1)[0].strip()
    # internal file opening function
    def _open(fp):
        # only consider .json or a file formatted (e.g. .txt) with a list of newline-separated strings
        if fp.extension == ".json":
            with fp.open() as f:
                return _sort(json.load(f))
        elif fp.extension in [".yaml", ".yml"]:
            with fp.open() as f:
                return _sort(yaml.safe_load(f))
        else:
            return _sort([_uncmt(l) for l in fp.read_text().split("\n") if _uncmt(l) != ""])
    # first, get the group (simply use exe_format if it is precisely a group)
    data, group = {}, get_format_group(exe_format)
    c1, c2 = config.default('data'), config['data']
    data_folders = [c1] if c1.is_samepath(c2) else [c1, c2]
    for cfg in data_folders:
        # consider most specific data first
        if group != exe_format:
            path = cfg.joinpath(_name(group))
            if path.exists():
                for datafile in path.listdir(lambda p: p.extension in DATA_EXTENSIONS):
                    if datafile.stem.endswith("_" + _name(exe_format)):
                        data["_".join(datafile.stem.split("_")[:-1]).upper()] = _open(datafile)
        # then the files without specific mention in a subfolder of 'cfg' that matches a format class and finally the
        #  files without specific mention at the root of 'cfg'
        for path in [cfg.joinpath(_name(group)), cfg]:
            if path.exists():
                for datafile in path.listdir(lambda p: p.extension in DATA_EXTENSIONS):
                    if not datafile.stem.endswith("_" + _name(exe_format)):
                        c = datafile.stem.upper()
                        data[c] = _add(_open(datafile), data[c]) if c in data else _open(datafile)
    return data


def make_test_dataset(n_features, n_samples=100, n_redundant=0, random_state=42):
    """  """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    single_feature = False
    if n_features == 1:
        single_feature, n_features = True, 2  # make_classification does not work with n_features=1
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=n_redundant,
                               random_state=random_state)
    X, Xt, y, yt = train_test_split(X, y, stratify=y, random_state=random_state)
    if single_feature:
        X, Xt = X[:,0] + max(X[:,0]), Xt[:,0] + max(Xt[:,0])  # ensure positive numbers too
        X, Xt = np.array([[x] for x in X]), np.array([[x] for x in Xt])
    return X, y, Xt, yt


def reduce_data(X, n_components=20, perplexity=30, random_state=42, imputer_strategy="mean", scaler=None,
                reduction_algorithm="PCA", return_scaled=False, return_suffix=False, return_meta=False, **kw):
    """ Reduce input data to 2 components, using a combination of PCA/ICA and TSNE, first imputing missing values and
         scaling data with an input scaler. """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    l = kw.get('logger', null_logger)
    n, n_cols, p, ra = n_components, len(X.columns), perplexity, reduction_algorithm
    rs, suffix, metadata = 42, "", {}
    # impute missing values and scale data
    l.debug(f"imputing values using strategy {imputer_strategy}...")
    X = SimpleImputer(missing_values=np.nan, strategy=imputer_strategy).fit_transform(X)
    l.debug(f"scaling data with {(scaler or MinMaxScaler).__name__}...")
    Xs = X = (scaler or MinMaxScaler)().fit_transform(X)
    # preprocess data with a PCA with n components to reduce the high dimensionality (better performance)
    if n < n_cols:
        from sklearn.decomposition import FastICA, PCA
        a = {'ICA': FastICA, 'PCA': PCA}[ra](n, random_state=rs() if is_function(rs) else rs)
        suffix += f"_{ra.lower()}{n}"
        metadata[ra + ':n'] = n
        l.debug(f"reducing data to {n} components with {ra}...")
        if 'target' in kw:
            a.fit(X, kw['target'])
            X = a.transform(X)
        else:
            X = a.fit_transform(X)
    # now reduce the n components to 2 dimensions with t-SNE (better results but less performance) if relevant
    if n == 2 and p != 30:
        l.warning("setting the perplexity when using PCA with 2 components has no effect because TSNE is not used")
    if n > 2:
        from sklearn.manifold import TSNE
        l.debug(f"reducing data to 2 components with TSNE...")
        X = TSNE(2, random_state=rs() if is_function(rs) else rs, perplexity=p).fit_transform(X)
        suffix += f"_tsne2-p{p}"
        metadata['TSNE:n'] = 2
        metadata['perplexity'] = p
    # return result(s) accordingly
    r = (X, )
    if return_scaled:
        r += (Xs, )
    if return_suffix:
        r += (suffix, )
    if return_meta:
        r += (metadata, )
    return r[0] if len(r) == 1 else r

