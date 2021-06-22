#find_library (CGNS_LIBRARIES NAMES libcgns.so PATHS $ENV{CGNS_LIBRARY_DIR} $ENV{HOME}/oss/lib NO_DEFAULT_PATH)
#find_path (CGNS_INCLUDE_DIRS "cgnslib.h" PATHS $ENV{CGNS_INCLUDE_DIR} $ENV{HOME}/oss/include NO_DEFAULT_PATH)
find_library (CGNS_LIBRARIES NAMES libcgns.so PATHS $ENV{CGNS_LIBRARY_DIR} $ENV{HOME}/oss/lib)
find_path (CGNS_INCLUDE_DIRS "cgnslib.h" PATHS $ENV{CGNS_INCLUDE_DIR} $ENV{HOME}/oss/include)
