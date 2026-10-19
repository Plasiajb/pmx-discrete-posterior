# Security

Use trusted source and release artifacts, verify the accompanying checksums,
and install in a dedicated environment. Model callbacks are executable Python
code with the permissions of the calling process; this package is not a sandbox.
Do not execute a model or example from an untrusted source.

## Report A Vulnerability

Use GitHub private vulnerability reporting through the repository's Security
tab if it is available to you. Otherwise, contact a repository maintainer
through an existing private channel to arrange a confidential report. Do not
post exploit details, credentials or patient data in an ordinary issue.

Include the affected version or commit, a minimal synthetic reproduction,
the impact and relevant environment details. No dedicated security email,
response deadline or guaranteed backport policy is published.

Ordinary bugs and numerical discrepancies can be reported through the issue
process in [CONTRIBUTING.md](CONTRIBUTING.md). Remove sensitive data first.

## Scientific Use

A successful calculation indicates that the implemented numerical checks
passed for the supplied inputs. It does not certify a dosing history,
clinical safety or the correctness of a user-supplied model. This research
package is not a standalone clinical decision-support tool.
