# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.18.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.18.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/vikasvarma/Documents/Development/imageocv/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/vikasvarma/Documents/Development/imageocv/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/imageocv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imageocv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imageocv.dir/flags.make

CMakeFiles/imageocv.dir/src/findlane.cpp.o: CMakeFiles/imageocv.dir/flags.make
CMakeFiles/imageocv.dir/src/findlane.cpp.o: ../src/findlane.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/vikasvarma/Documents/Development/imageocv/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imageocv.dir/src/findlane.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imageocv.dir/src/findlane.cpp.o -c /Users/vikasvarma/Documents/Development/imageocv/cpp/src/findlane.cpp

CMakeFiles/imageocv.dir/src/findlane.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imageocv.dir/src/findlane.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vikasvarma/Documents/Development/imageocv/cpp/src/findlane.cpp > CMakeFiles/imageocv.dir/src/findlane.cpp.i

CMakeFiles/imageocv.dir/src/findlane.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imageocv.dir/src/findlane.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vikasvarma/Documents/Development/imageocv/cpp/src/findlane.cpp -o CMakeFiles/imageocv.dir/src/findlane.cpp.s

CMakeFiles/imageocv.dir/src/imageio.cpp.o: CMakeFiles/imageocv.dir/flags.make
CMakeFiles/imageocv.dir/src/imageio.cpp.o: ../src/imageio.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/vikasvarma/Documents/Development/imageocv/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/imageocv.dir/src/imageio.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imageocv.dir/src/imageio.cpp.o -c /Users/vikasvarma/Documents/Development/imageocv/cpp/src/imageio.cpp

CMakeFiles/imageocv.dir/src/imageio.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imageocv.dir/src/imageio.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vikasvarma/Documents/Development/imageocv/cpp/src/imageio.cpp > CMakeFiles/imageocv.dir/src/imageio.cpp.i

CMakeFiles/imageocv.dir/src/imageio.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imageocv.dir/src/imageio.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vikasvarma/Documents/Development/imageocv/cpp/src/imageio.cpp -o CMakeFiles/imageocv.dir/src/imageio.cpp.s

# Object files for target imageocv
imageocv_OBJECTS = \
"CMakeFiles/imageocv.dir/src/findlane.cpp.o" \
"CMakeFiles/imageocv.dir/src/imageio.cpp.o"

# External object files for target imageocv
imageocv_EXTERNAL_OBJECTS =

libimageocv.a: CMakeFiles/imageocv.dir/src/findlane.cpp.o
libimageocv.a: CMakeFiles/imageocv.dir/src/imageio.cpp.o
libimageocv.a: CMakeFiles/imageocv.dir/build.make
libimageocv.a: CMakeFiles/imageocv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/vikasvarma/Documents/Development/imageocv/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libimageocv.a"
	$(CMAKE_COMMAND) -P CMakeFiles/imageocv.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imageocv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imageocv.dir/build: libimageocv.a

.PHONY : CMakeFiles/imageocv.dir/build

CMakeFiles/imageocv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imageocv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imageocv.dir/clean

CMakeFiles/imageocv.dir/depend:
	cd /Users/vikasvarma/Documents/Development/imageocv/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/vikasvarma/Documents/Development/imageocv/cpp /Users/vikasvarma/Documents/Development/imageocv/cpp /Users/vikasvarma/Documents/Development/imageocv/cpp/build /Users/vikasvarma/Documents/Development/imageocv/cpp/build /Users/vikasvarma/Documents/Development/imageocv/cpp/build/CMakeFiles/imageocv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imageocv.dir/depend
