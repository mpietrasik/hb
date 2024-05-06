% Compiles sampler using Boost 1.67.0 C++ libraries
%
% If compiling for 64-bit MATLAB, boost should be compiled 
% with "bjam address-model=64" to produce 64-bit libraries.

BOOST_PATH      = 'C:/Users/marci/boost_1_67_0_old_pc';            
BOOST_LIB_PATH  = 'C:/Users/marci/boost_1_67_0_old_pc/stage/lib'; 

eval(['mex COMPFLAGS="$COMPFLAGS -I' BOOST_PATH '" LINKFLAGS="$LINKFLAGS /LIBPATH:' BOOST_LIB_PATH '" gateway.cpp'])

clear BOOST_PATH BOOST_LIB_PATH;
