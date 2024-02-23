from setuptools import find_packages, setup

package_name = "tbro"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy >= 1.19.5",
        "scipy >= 1.5.4",
        "torch >= 1.10.1",
        "tensorboard >= 2.10.1",
    ],
    zip_safe=True,
    maintainer="parallels",
    maintainer_email="parallels@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["tbro_node = tbro.tbro_node:main"],
    },
)
