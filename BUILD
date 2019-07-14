# For blaze on Google's infrastructure.

py_library(
    name = "xiuminglib",
    srcs = glob(["xiuminglib/**/*.py"]),
    data = glob(["data/**/*"]),
    srcs_version = "PY3",
    deps = [
        # bpy not in third_party, so has to be wrapped into a MPM package
        "//pyglib:gfile",
        "//third_party/py/cvx2",
        "//third_party/py/matplotlib",
        "//third_party/py/mpl_toolkits/axes_grid1",
        "//third_party/py/mpl_toolkits/mplot3d",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        # FIXME: Listing OpenEXR and Imath as deps here segfaults bpy on Borg
        # "//third_party/py/OpenEXR",
        # "//third_party/py/Imath",
        "//third_party/py/tqdm",
    ],
)
