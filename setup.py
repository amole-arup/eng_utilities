from setuptools import find_packages, setup

setup(
    name='eng_utilities',
    version='0.1.0',
    description='General Engineering Tools Repository',
    author='andrew.mole@arup.com',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.1.1'],
    test_suite='tests',
    url='https://github.com/amole-arup/eng_utilities',
    packages=find_packages(include=['eng_utilities'])
)