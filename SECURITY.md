# Security Policy

Semantic Search Platform accepts documents, queries, and structured metadata. Injection, unsafe deserialization, oversized-input bypasses, cross-document exposure, path handling, or denial-of-service issues should be reported privately.

Use GitHub's **Security → Report a vulnerability** flow when available. Otherwise, contact the maintainer through the GitHub profile. Include sanitized inputs, affected configuration, impact, and a minimal reproduction.

The current `main` branch is supported. Internet-facing deployments must add authentication, authorization, TLS, tenant isolation, request throttling, and production secret management.
