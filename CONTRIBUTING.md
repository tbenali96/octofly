# Contributing

## commit naming

"<commit-type>: <action verb><description>"

### Available type

- FEAT: add a feature
- FIX: fix a bug
- CONF: change configuration
- DOC: add documentation
- REFACTO: refacto the code

## convention of code

- The variables, the functions and the code are in English 
- The logging messages are in French 
- PEP8 code formatting
- type hinting (checked by mypy)
- if POO, no extra step besides attributes initialization in class constructor
- test name should be like "test_{function_name}__{test_description}"
- use f string
- no magic string or number
- docstring for each function (at least important one)
- use pathlib everywhere

## quality convention

- code coverage above 85%

## git workflow

- create a new branch
- commit modification in my new branch
- rebase every day
- squash useless commit (typo, flake8, ...)
- follow commit name convention
- create a PR when finished
- warn the team in Teams that in new PR is ready for review
- rebase every day
- when PR is validated, merge it

## Versionning

 The saft_formation package is versioned using 3 digits: x.y.z

- increase z in case of bug fixes, refacto or documentation which don't modify any behavior
- increase y in case new features which don't add new behavior without modify previous behaviors
- increase x in case of compatibility breaking new features
