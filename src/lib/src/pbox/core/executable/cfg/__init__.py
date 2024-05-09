# -*- coding: UTF-8 -*-
from .__common__ import *
from .__common__ import __all__


class FeatureExtractionFunctions:

    @staticmethod
    def get_api_features(cfg, mal_apis, non_apis={'PathTerminator'}, suffixes={'', 'Ex', 'A', 'ExA'}):
        """Extracts features based on api imports and use

        Args:
            cfg: a CFG object
            mal_apis: a list of strings of malicious api names
        KwArgs:
            non_apis: a set of Angr zero-sized node names that are non-apis
            suffixes: a set of suffixed that can be appended to mal_apis, resulting in (possible) similar implementations of those apis
        Returns:
            api_uniqmalused_uniqtotused_ratio: ratio of (unique malicious api functions used) / (unique api functions used)
            api_malused_totused_ratio: ratio of (total malicious api functions used) / (total api functions used) [Note: doesn't take into account that loops can cause some of these nodes to be traversed more than others]
            api_uniqmalused_totimp_ratio: ratio of (unique malicious api functions used) / (api functions imported)
            api_malimp_totimp_ratio: ratio of (malicious api functions imported) / (api functions imported) [also implemented in pefeats, but with fixed set of mal_apis, while here it is a function parameter]
            api_malapis_present: list of booleans indicating whether the mal_api (or a similar version with a suffix) is imported or not
        """
        imported_apis = cfg.model.project.loader.main_object.imports.keys()
        if suffixes:
            stripped_apis = []
            non_empty_suffixes = {s for s in suffixes if s != ''}
            for api in imported_apis:
                for suffix in non_empty_suffixes:
                    if api.endswith(suffix):
                        api = api[:-len(suffix)]
                        break
                stripped_apis.append(api)
            imported_apis = stripped_apis
        mal_apis_set = set(mal_apis)
        mal_imported_apis = [api for api in imported_apis if api in mal_apis_set]
        used_apis = [n.name for n in cfg.model.nodes() if n.name is not None and n.name not in non_apis]
        unique_used_apis = set(used_apis)
        mal_used_apis = [api for api in used_apis if api in mal_apis_set]
        unique_mal_used_apis = set(mal_used_apis)
        api_uniqmalused_uniqtotused_ratio = api_malused_totused_ratio = -1 # Default value if no used apis
        api_uniqmalused_totimp_ratio = api_malimp_totimp_ratio = -1 # Default value if no imported_apis
        if unique_used_apis:
            api_uniqmalused_uniqtotused_ratio = len(unique_mal_used_apis) / len(unique_used_apis)
            api_malused_totused_ratio = len(mal_used_apis) / len(used_apis)
        if imported_apis:
            api_uniqmalused_totimp_ratio = len(unique_mal_used_apis) / len(imported_apis)
            api_malimp_totimp_ratio = len(mal_imported_apis) / len(imported_apis)
        api_malapis_present = [api in imported_apis for api in mal_apis]
        return api_uniqmalused_uniqtotused_ratio, api_malused_totused_ratio, api_uniqmalused_totimp_ratio, api_malimp_totimp_ratio, api_malapis_present

