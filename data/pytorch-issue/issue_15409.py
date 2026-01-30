if isinstance(py_ast.body[0], ast.ClassDef):
    module_name = source[source.index('class') + 6 : source.index('(')]
    raise RuntimeError("cannot create a module (%s) inside jit.script" % module_name)