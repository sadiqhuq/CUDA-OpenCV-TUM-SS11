
win32 {
  TEMPLATE = vcapp
  CONFIG += debug console
}
unix {
  TEMPLATE = app
  CONFIG += qt debug warn_on 
}

TARGET = diffusion
DEPENDPATH +=


INCLUDEPATH += $(OPENCVDIR)/include
INCLUDEPATH += $(OPENCVDIR)/include/opencv
INCLUDEPATH += $(OPENCVDIR)/include/opencv2
INCLUDEPATH += $(OPENCVDIR)/modules/core/include
INCLUDEPATH += $(OPENCVDIR)/modules/calib3d/include
INCLUDEPATH += $(OPENCVDIR)/modules/contrib/include
INCLUDEPATH += $(OPENCVDIR)/modules/features2d/include
INCLUDEPATH += $(OPENCVDIR)/modules/flann/include
INCLUDEPATH += $(OPENCVDIR)/modules/gpu/include
INCLUDEPATH += $(OPENCVDIR)/modules/haartraining/include
INCLUDEPATH += $(OPENCVDIR)/modules/highgui/include
INCLUDEPATH += $(OPENCVDIR)/modules/imgproc/include
INCLUDEPATH += $(OPENCVDIR)/modules/java/include
INCLUDEPATH += $(OPENCVDIR)/modules/legacy/include
INCLUDEPATH += $(OPENCVDIR)/modules/ml/include
INCLUDEPATH += $(OPENCVDIR)/modules/objdetect/include
INCLUDEPATH += $(OPENCVDIR)/modules/video/include


#--- additional preprocessor directives ---
win32 {
  DEFINES += _CONSOLE
  DEFINES += _DEBUG
  DEFINES += NOMINMAX
}


#--- Input files ---
HEADERS += diffusion.cuh 

SOURCES += main.cpp 

#CUDA_HEADERS += 
CUDA_SOURCES += diffusion.cu 


########################################################################
#  CUDA
########################################################################
win32 {
 INCLUDEPATH += $(CUDA_INC_DIR)
 INCLUDEPATH += $(CUDA_SDK_INC_DIR)
 QMAKE_LIBDIR += $(CUDA_LIB_DIR)
 QMAKE_LIBDIR += $(CUDA_SDK_LIB_DIR)
 QMAKE_LIBDIR += $(OPENCVDIR)/build/lib
 LIBS += -lcudart -lcuda
 LIBS += -lopencv_core
 LIBS += -lopencv_highgui
 LIBS += -lopencv_imgproc

 cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
 cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 #cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -ccbin '"$(VCInstallDir)bin"' -c -D_DEBUG -DWIN32 -D_CONSOLE -Xcompiler $$join(QMAKE_CXXFLAGS,",") -I'"./ "' $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o "$(ConfigurationName)"${QMAKE_FILE_OUT}
}

unix {
 # auto-detect CUDA path
 #CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
 # set CUDA path manually
 CUDA_DIR = $(CUDADIR)
 INCLUDEPATH += $$CUDA_DIR/include
 INCLUDEPATH += $(CUDA_SDK_INC_DIR)

 # Compiler flags tuned for my system
 #QMAKE_CXXFLAGS += -I../ -O99 -pipe -g -Wall
 QMAKE_LIBDIR += $$CUDA_LIB_DIR
 QMAKE_LIBDIR += $(CUDADRVDIR)
 LIBS += -L$(CUDA_LIB_DIR) -L$(CUDA_SDK_LIB_DIR) -lcuda -lcudart
 QMAKE_LIBDIR += $(OPENCVDIR)/build/lib
 LIBS += -lopencv_core
 LIBS += -lopencv_highgui
 LIBS += -lopencv_imgproc


 cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
 cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH," -I ",-I ",") ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 #cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH," -I ",-I ",") ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
 cuda.dependcy_type = TYPE_C
 cuda.depend_command = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH," -I ",-I ",") ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_UNIX_COMPILERS += cuda
########################################################################