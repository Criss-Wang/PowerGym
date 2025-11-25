Contributing
============

We welcome contributions to PowerGrid!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Submit a pull request

Development Setup
-----------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/powergrid.git
   cd powergrid

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in development mode
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=powergrid --cov-report=html

   # Run specific test file
   pytest tests/test_agents.py

Code Style
----------

PowerGrid follows PEP 8 style guidelines:

- Use ``black`` for code formatting
- Use ``isort`` for import sorting
- Use ``flake8`` for linting
- Maximum line length: 100 characters

.. code-block:: bash

   # Format code
   black powergrid tests

   # Sort imports
   isort powergrid tests

   # Check style
   flake8 powergrid tests

Documentation
-------------

Update documentation when adding features:

.. code-block:: bash

   # Build docs locally
   cd docs/source
   make html

   # View docs
   open _build/html/index.html

Write clear docstrings following NumPy style:

.. code-block:: python

   def my_function(param1, param2):
       """
       Short description.

       Longer description with details.

       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type
           Description of param2

       Returns
       -------
       type
           Description of return value
       """
       pass

Pull Request Guidelines
-----------------------

- Write clear, descriptive commit messages
- Include tests for new features
- Update documentation
- Ensure all tests pass
- Keep PRs focused on a single feature/fix

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   <type>: <subject>

   <body>

   <footer>

Types: ``feat``, ``fix``, ``docs``, ``style``, ``refactor``, ``test``, ``chore``

Example:

.. code-block:: text

   feat: Add ADMM protocol for distributed optimization

   Implement Alternating Direction Method of Multipliers (ADMM)
   as a new vertical protocol for coordinating devices.

   Closes #123

Reporting Issues
----------------

Report bugs and request features on GitHub Issues:

- Use a clear, descriptive title
- Provide detailed reproduction steps
- Include environment information (OS, Python version, etc.)
- Attach relevant logs or error messages

Community
---------

- GitHub Discussions: Ask questions and share ideas
- Slack: Join our community chat (link in README)
- Email: Contact maintainers at powergrid@example.com

Code of Conduct
---------------

Be respectful and inclusive. We follow the Contributor Covenant Code of Conduct.
