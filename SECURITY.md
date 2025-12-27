# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in ConsensusMind, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues via email to:

**Email:** [your-security-email@dslabs.network]

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)

### Response Timeline

- **Initial Response:** Within 48 hours of report
- **Status Update:** Within 7 days
- **Fix Timeline:** Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: Next release cycle

### Disclosure Policy

- Security issues will be fixed before public disclosure
- Reporter will be credited (unless anonymity is requested)
- Public disclosure will be coordinated with the reporter
- CVE will be requested for critical vulnerabilities

### Security Best Practices

When using ConsensusMind:

1. **API Keys:** Never commit API keys or secrets to the repository
2. **Configuration:** Use environment variables for sensitive data
3. **Updates:** Keep dependencies updated regularly
4. **LLM Endpoints:** Only connect to trusted inference endpoints
5. **PDF Processing:** Be cautious when processing untrusted PDFs

### Scope

Security issues we accept:
- Remote code execution
- Authentication/authorization bypass
- Information disclosure
- Injection vulnerabilities (SQL, command, etc.)
- Cryptographic vulnerabilities

Out of scope:
- Denial of Service from rate limiting
- Issues requiring physical access
- Social engineering attacks
- Issues in third-party dependencies (report to them directly)

## Security Features

ConsensusMind implements:
- HTTPS for all external API calls
- Input validation and sanitization
- Rate limiting for external services
- Secure dependency management
- No execution of untrusted code

## Contact

For general security questions: [security@dslabs.network]

For urgent security issues: [security@dslabs.network]

---

Thank you for helping keep ConsensusMind and our users safe.
