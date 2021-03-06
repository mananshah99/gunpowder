from setuptools import setup

setup(
        name='gunpowder',
        version='0.2',
        description='Data loading DAG for Greentea.',
        url='https://github.com/funkey/gunpowder',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=[
            'gunpowder',
            'gunpowder.nodes',
            'gunpowder.caffe',
            'gunpowder.caffe.nodes',
            'gunpowder.ext',
            'gunpowder.tests',
            'gunpowder.tests.cases',
        ],
        install_requires = [
            "numpy",
            "scipy",
            "h5py",
            "scikit-image",
        ],
)
