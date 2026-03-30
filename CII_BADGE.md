# CII Best Practices Badge Guide

This project aims to achieve the [OpenSSF CII Best Practices Badge](https://bestpractices.coreinfrastructure.org/) to demonstrate commitment to security and quality.

## Current Status

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/badge_placeholder)](https://bestpractices.coreinfrastructure.org/projects/badge_placeholder)

*Note: Replace `badge_placeholder` with your actual project ID after applying.*

## How to Apply

### Step 1: Create an Account

1. Go to <https://bestpractices.coreinfrastructure.org/>
2. Sign in with your GitHub account
3. Click "Add a Project"

### Step 2: Submit Your Project

Fill in the required information:
- **Project URL**: `https://github.com/PIsberg/claude_light`
- **Project Homepage**: Same as above
- **Description**: "Claude Light is an interactive CLI chat tool for querying and editing codebases using Claude with hybrid RAG and prompt caching"

### Step 3: Complete the Questionnaire

The badge has three tiers: **Passing**, **Silver**, and **Gold**. Each tier requires meeting more criteria.

#### Passing Tier (Required)

Key criteria we already meet:
- ✅ **License**: LICENSE file present (PolyForm Noncommercial)
- ✅ **Security Policy**: SECURITY.md with vulnerability reporting process
- ✅ **Versioning**: Uses semantic versioning via Git tags
- ✅ **Change Control**: Git history with commit messages
- ✅ **Code Review**: Branch protection requires PR reviews
- ✅ **CI/CD**: GitHub Actions runs tests on every push
- ✅ **Static Analysis**: CodeQL runs on every push
- ✅ **Dependencies**: Dependabot keeps dependencies updated
- ✅ **Fuzz Testing**: Hypothesis property-based testing enabled

#### Silver Tier (Target)

Additional criteria to work towards:
- [ ] **Documentation**: More comprehensive user documentation
- [ ] **Test Coverage**: >70% code coverage
- [ ] **Security Audit**: External security review
- [ ] **Hardening**: Compiler/linker flags for C extensions (if applicable)

#### Gold Tier (Aspirational)

Advanced criteria:
- [ ] **Formal Verification**: Formal methods or proofs
- [ ] **Multiple Independent Audits**: More than one security audit
- [ ] **Reproducible Builds**: Bit-for-bit reproducible builds

### Step 4: Submit for Review

Once you've completed the questionnaire:
1. Review your answers carefully
2. Submit for community review
3. Address any feedback from reviewers

### Step 5: Update the Badge

After approval, update the badge URL in README.md:

```markdown
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/PROJECT_ID/badge)](https://bestpractices.coreinfrastructure.org/projects/PROJECT_ID)
```

Replace `PROJECT_ID` with your assigned project number.

## Maintaining the Badge

- **Annual Recertification**: You must reconfirm annually that the project still meets criteria
- **Report Changes**: If the project changes significantly, update your submission
- **Monitor Email**: Watch for recertification reminders from CII

## Benefits

- **Trust Signal**: Shows users you take security seriously
- **Best Practices**: Encourages following industry standards
- **Community Recognition**: Listed on OpenSSF website
- **Scorecard Points**: Contributes to OpenSSF Scorecard score

## Resources

- [CII Best Practices Program](https://bestpractices.coreinfrastructure.org/)
- [Criteria Details](https://bestpractices.coreinfrastructure.org/en/criteria)
- [FAQ](https://bestpractices.coreinfrastructure.org/en/faq)
