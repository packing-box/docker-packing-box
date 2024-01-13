# -*- coding: UTF-8 -*-
from tinyscript import code, logging, re, time
from tinyscript.helpers import Capture, Timeout, TimeoutError


def _config_angr_loggers():
    angr_loggers = ["angr.*", "cle\..*", "pyvex.*"]
    configure_logging(reset=True, exceptions=angr_loggers)
    from ...helpers.config import _LOG_CONFIG
    for l in logging.root.manager.loggerDict:
        if any(re.match(al, l) for al in angr_loggers):
            logging.getLogger(l).setLevel([logging.WARNING, logging.DEBUG][_LOG_CONFIG[0]])
    from cle.backends.pe.regions import PESection
    code.insert_line(PESection.__init__, "from tinyscript.helpers import ensure_str", 0)
    code.replace(PESection.__init__, "pe_section.Name.decode()", "ensure_str(pe_section.Name)")
lazy_load_module("angr", postload=_config_angr_loggers)


class CFG:
    logger = logging.getLogger("executable.cfg")
    
    def __init__(self, target, engine=None, **kw):
        self.__target, self.engine = str(target), engine
    
    def compute(self, algorithm=None, timeout=None, **kw):
        l = self.__class__.logger
        if self.__project is None:
            l.error(f"{self.__target}: CFG project not created ; please set the engine")
            return
        try:
            with Capture() as c, Timeout(timeout or config['extract_timeout'], stop=True) as to:
                getattr(self.__project.analyses, f"CFG{algorithm or config['extract_algorithm']}") \
                    (fail_fast=False, resolve_indirect_jumps=True, normalize=True, model=self.model)
        except TimeoutError:
            l.warning(f"{self.__target}: Timeout reached when extracting CFG")
        except Exception as e:
            l.error(f"{self.__target}: Failed to extract CFG")
            l.exception(e)
    
    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, name=None):
        name = config['angr_engine'] if name is None else name
        cls = "UberEngine" if name in ["default", "vex"] else f"UberEngine{name.capitalize()}"
        try:
            self._engine = getattr(angr.engines, cls)
            self.__project = angr.Project(self.__target, load_options={'auto_load_libs': False}, engine=self._engine)
            self.model = self.__project.kb.cfgs.new_model(f"{self.__target}")
        except Exception as e:
            self._engine = self.__project = None
            self.__class__.logger.exception(e)

