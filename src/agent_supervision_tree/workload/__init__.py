"""Reference workload: Ingestor -> ExtractorPool -> Validator -> WebhookSink under supervision.

The workload exercises every runtime control with synthetic documents and a seeded,
fault-injecting fake model. :mod:`agent_supervision_tree.workload.pipeline` builds the tree and
:mod:`agent_supervision_tree.workload.demo` runs it offline with faults injected.
"""
