import sys
from uuid import uuid4

from typing import Sequence, Tuple, Any, TypeVar, Dict

from functools import partialmethod

def partialclass(name, cls, *args, **kwds):
	new_cls = type(name, (cls,), {
		'__init__': partialmethod(cls.__init__, *args, **kwds),
		'__bound_args__': args,
		'__bound_kwargs__': kwds,
	})
	try:
		new_cls.__module__ = sys._getframe(1).f_globals.get(
			'__name__', '__main__'
		)
	except (AttributeError, ValueError):
		pass
	return new_cls

class DependencyInjector():
	def __init__(
		self,
		components : Sequence[Tuple[TypeVar,Sequence[Any],Dict]] = []
	):
		self._bindings = {}
		self.uuid = uuid4()
		for cls,args,kwargs in components:
			self._register(cls,*args,**kwargs)

	def _register(self, cls, *args, **kwargs):
		"""Create and register a dependency-injected version of cls"""
		name = f"{cls.__name__}_injected_{self.uuid}"
		injected = partialclass(name, cls, *args, **kwargs)
		self._bindings[cls] = injected

	def inject(self, cls):
		return self._bindings.get(cls, cls)
