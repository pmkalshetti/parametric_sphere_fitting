include(RegexUtils)
test_escape_string_as_regex()

# Please let's not glob these

install(FILES
	COPYING.BSD
	COPYING.GPL
	COPYING.LGPL
	COPYING.MINPACK
	COPYING.MPL2
	COPYING.README
	CTestConfig.cmake
	INSTALL
	README.md
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel
  )

add_subdirectory(src)
