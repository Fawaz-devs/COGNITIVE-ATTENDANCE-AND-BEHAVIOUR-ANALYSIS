#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dlib::dlib" for configuration "RelWithDebInfo"
set_property(TARGET dlib::dlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(dlib::dlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C;CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/dlib19.24.99_relwithdebinfo_64bit_msvc1941.lib"
  )

list(APPEND _cmake_import_check_targets dlib::dlib )
list(APPEND _cmake_import_check_files_for_dlib::dlib "${_IMPORT_PREFIX}/lib/dlib19.24.99_relwithdebinfo_64bit_msvc1941.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
