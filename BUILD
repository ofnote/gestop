licenses(["notice"])  # Apache 2.0

#package(default_visibility = ["//mediapipe/examples:__subpackages__"])
package(default_visibility = ["//visibility:public"])

# Linux only
cc_binary(
    name = "hand_tracking_gpu",
    deps = [
        "//gestop:hand_tracking_landmarks",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
        ":zeromq",
    ],
)

cc_binary(
    name = "hand_tracking_cpu",
    deps = [
        "//gestop:hand_tracking_landmarks_cpu",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        ":zeromq",
    ],
)

cc_proto_library(
    name = "landmarkList_cc_proto",
    deps = [":landmarkList_proto"],
)

proto_library(
    name = "landmarkList_proto",
    srcs = ["proto/landmarkList.proto"],
)


cc_import(
    name = "zeromq",
    hdrs = ["zmq.hpp"],
    shared_library = "libzmq.so",
)

#cc_library(
#    name = "zeromq",
#    #hdrs = ["zmq.hpp"],
#    hdrs = glob(["*.hpp"]),
#    visibility = ["//visibility:public"],
#    srcs = ["libzmq.so"]

#    includes = ["zmq.hpp"],
#    #shared_library = "libzmq.so",
#    srcs = [ 
#        "cppzmq/zmq.hpp",
#        "libzmq.so"],
#)

# Linux only.
# Must have a GPU with EGL support:
# ex: sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
# (or similar nvidia/amd equivalent)
cc_library(
    name = "hand_tracking_landmarks",
    srcs = ["hand_tracking_landmarks.cc"],
    deps = [
        ":landmarkList_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
    ],
)


cc_library(
    name = "hand_tracking_landmarks_cpu",
    srcs = ["hand_tracking_landmarks_cpu.cc"],
    deps = [
        ":landmarkList_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)
