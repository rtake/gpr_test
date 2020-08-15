# reference = http://urin.github.io/posts/2013/simple-makefile-for-clang
TARGET = ./bin/$(shell basename `readlink -f .`)

CXX = g++
CXXFLAGS = -g
# LDFLAGS =
INCLUDE = -I/usr/include -I/home/rtake/eigen-3.3.7 
# LIBS = /home/rtake/eigen-3.3.7

TARGET = ./bin/$(shell basename `readlink -f .`)

SRCDIR = ./src
SOURCES   = $(wildcard $(SRCDIR)/*.cpp)

OBJDIR    = ./obj
OBJECTS   = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.cpp=.o)))


$(TARGET): $(OBJECTS) $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o :$(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<


clean:
	rm -f $(TARGET) $(OBJECTS)