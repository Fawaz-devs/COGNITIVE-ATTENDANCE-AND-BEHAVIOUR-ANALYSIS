#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dlib::dlib" for configuration "MinSizeRel"
set_property(TARGET dlib::dlib APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(dlib::dlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "C;CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/dlib19.24.99_minsizerel_64bit_msvc1941.lib"
  )

list(APPEND _cmake_import_check_targets dlib::dlib )
list(APPEND _cmake_import_check_files_for_dlib::dlib "${_IMPORT_PREFIX}/lib/dlib19.24.99_minsizerel_64bit_msvc1941.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
