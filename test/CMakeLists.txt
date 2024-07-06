include(make_unit)
set(unit_root "${CMAKE_SOURCE_DIR}/test")

glob_unit(${unit_root} "generator.cpp")
glob_unit(${unit_root} "predicates.cpp")
glob_unit(${unit_root} "numeric.cpp")
glob_unit(${unit_root} "copy.cpp")
glob_unit(${unit_root} "reduce.cpp")
glob_unit(${unit_root} "transform.cpp")