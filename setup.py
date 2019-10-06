from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.readlines()


setup(name='skprofiling',
      version='0.1',
      description='Profiling of machine learning models based on scikit-learning interface',
      long_description=readme(),
      keywords='pandas machine learning profiling model',
      url='https://github.com/tiagohcalves/scikit-profiling',
      author='Tiago Alves',
      author_email='tiagohcalves@gmail.com',
      license='MIT',
      packages=['skprofiling'],
      install_requires=requirements(),
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
