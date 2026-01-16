Contributing
============

We welcome contributions to CR_Score!

How to Contribute
-----------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/edmunlee87/CR_Score.git
   cd CR_Score
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   pytest tests/unit -v
   pytest tests/unit -v --cov=src/cr_score

Code Style
----------

We use:

- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

.. code-block:: bash

   black src/cr_score
   flake8 src/cr_score
   mypy src/cr_score

Documentation
-------------

Build documentation:

.. code-block:: bash

   cd docs
   make html

Areas for Contribution
----------------------

- Additional visualization types
- New model types
- Performance optimizations
- Documentation improvements
- Bug fixes

License
-------

By contributing, you agree to license your contributions under the same license as the project.
