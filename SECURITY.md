# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < latest| :x:                |

We recommend always using the latest release for security updates.

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please follow these steps:

### Where to Report

**Do not open a public issue.** Instead, report vulnerabilities using one of these methods:

1. **GitHub Private Vulnerability Reporting** (preferred):
   - Go to the [Security](https://github.com/PIsberg/claude_light/security) tab
   - Click "Report a vulnerability"
   - Provide details about the vulnerability

2. **Email**: Send an email to [isberg.peter+cl@gmail.com](mailto:isberg.peter+cl@gmail.com) with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (if applicable)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Investigation**: We will investigate the report and confirm the vulnerability
- **Timeline**: We aim to resolve critical vulnerabilities within 7 days
- **Communication**: We will keep you informed of our progress throughout the process
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Coordinated Disclosure

Once a vulnerability is fixed, we will:
1. Publish a security advisory on GitHub
2. Release a patched version
3. Notify users via release notes

## Security Best Practices

This project follows these security practices:

- **Dependency Management**: Automated dependency updates via Dependabot
- **Code Review**: All changes require review before merging
- **CI/CD Security**: GitHub Actions workflows use pinned dependencies and minimal permissions
- **Static Analysis**: CodeQL runs on every push and PR
- **SBOM**: Software Bill of Materials generated for transparency

## Known Security Contacts

- **Maintainer**: Peter Isberg
- **Email**: isberg.peter+cl@gmail.com
