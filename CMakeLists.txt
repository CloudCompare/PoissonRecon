cmake_minimum_required( VERSION 3.10 )

project( QPOISSON_RECON_LIB )

add_library( ${PROJECT_NAME} STATIC )
if( UNIX OR MINGW )
	target_compile_options( ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-else> )
elseif( MSVC )
	target_compile_options( ${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj> )
endif()

target_sources( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap/PoissonReconLib.h
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap/PoissonReconLib.cpp
)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/Src_CC_wrap
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/Src
)

# Use openMP for poisson else it default to std::async for threading
option( POISSON_RECON_WITH_OPEN_MP "Compile PoissonRecon plugin with OpenMP support" OFF )

if( POISSON_RECON_WITH_OPEN_MP )
	find_package( OpenMP )

	if( OpenMP_CXX_FOUND )
		target_link_libraries( ${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX )
	endif()
endif ()
