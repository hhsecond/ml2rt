from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

exec(open('ml2rt/version.py').read())

requirements = []
extras_require = {
    'tensorflow': ['tensorflow'],
    'pytorch': ['torch'],
    'sklearn': ['sklearn', 'skl2onnx', 'pandas', 'onnxmltools', 'onnxconverter_common', 'numpy'],
    'sparkml': ['pyspark', 'onnxmltools', 'onnxconverter_common', 'numpy'],
    'onnx': ['onnx'],
    'all': []
}

for packages in extras_require.values():
    extras_require['all'].extend(packages)

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Sherin Thomas",
    author_email='sherin@tensorwerk.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Machine learning utilities for model conversion, serialization, loading etc",
    install_requires=requirements,
    extras_require=extras_require,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='ml2rt',
    name='ml2rt',
    packages=find_packages(include=['ml2rt']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/hhsecond/ml2rt',
    version=__version__,  # comes from ml2rt/version.py
    zip_safe=False,
)
