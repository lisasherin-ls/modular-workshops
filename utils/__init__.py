"""Configure HTTPS clients to use the operating system trust store."""

import truststore

truststore.inject_into_ssl()
