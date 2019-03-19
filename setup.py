from setuptools import setup
 
setup(
    name='ascends-toolkit',
    version='0.1',
    description='ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists',
    long_description='',
    url='https://github.com/liza183/ascends-toolkit',
    author='Matt Sangkeun Lee',
    author_email='lee4@ornl.gov',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=[],
    py_modules=['clien'],
    python_requires='>=3.2',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'tensorflow',
        'keras',
        'scikit-learn',
        'minepy',
    ],  
    scripts=['train_regression.py','ascends_server.py','train_classifier.py','classify_with_model.py','predict_with_model.py']
)
