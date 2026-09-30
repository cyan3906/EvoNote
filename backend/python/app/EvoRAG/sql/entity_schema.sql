CREATE TABLE IF NOT EXISTS evorag_entities (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    canonical_name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(64) NOT NULL DEFAULT 'concept',
    aliases_json JSON NOT NULL,
    identity_description TEXT NOT NULL,
    summary TEXT NOT NULL,
    description_for_match TEXT NOT NULL,
    embedding_json JSON NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_evorag_entities_normalized_name (normalized_name),
    INDEX idx_evorag_entities_type_name (entity_type, normalized_name),
    INDEX idx_evorag_entities_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_entities_scope_name (workspace_id, project_id, collection_id, domain, normalized_name),
    INDEX idx_evorag_entities_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_attributes (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    entity_id BIGINT NOT NULL,
    attr_type VARCHAR(32) NOT NULL,
    value_text TEXT NOT NULL,
    value_fingerprint CHAR(64) NOT NULL,
    confidence DOUBLE NOT NULL DEFAULT 0.7,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_evorag_entity_attr_fingerprint (entity_id, attr_type, value_fingerprint),
    INDEX idx_evorag_entity_attributes_entity (entity_id),
    INDEX idx_evorag_entity_attributes_type (attr_type),
    CONSTRAINT fk_evorag_entity_attributes_entity
        FOREIGN KEY (entity_id) REFERENCES evorag_entities(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_attribute_evidence (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    attribute_id BIGINT NOT NULL,
    note_id VARCHAR(128) NOT NULL DEFAULT '',
    block_id VARCHAR(128) NOT NULL DEFAULT '',
    block_index INT NOT NULL DEFAULT -1,
    evidence_text TEXT NOT NULL,
    evidence_fingerprint CHAR(64) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_evorag_attr_evidence_fingerprint (attribute_id, evidence_fingerprint),
    INDEX idx_evorag_entity_evidence_attribute (attribute_id),
    INDEX idx_evorag_entity_evidence_block (note_id, block_id),
    CONSTRAINT fk_evorag_entity_evidence_attribute
        FOREIGN KEY (attribute_id) REFERENCES evorag_entity_attributes(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_resolution_audit (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    incoming_name VARCHAR(255) NOT NULL,
    incoming_type VARCHAR(64) NOT NULL DEFAULT '',
    decision VARCHAR(32) NOT NULL,
    matched_entity_id BIGINT NULL,
    score DOUBLE NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_evorag_resolution_audit_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_resolution_audit_name (incoming_name),
    INDEX idx_evorag_resolution_audit_decision (decision),
    CONSTRAINT fk_evorag_resolution_audit_entity
        FOREIGN KEY (matched_entity_id) REFERENCES evorag_entities(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_aliases (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    scope_hash CHAR(40) NOT NULL,
    entity_id BIGINT NOT NULL,
    alias_text VARCHAR(255) NOT NULL,
    normalized_alias VARCHAR(255) NOT NULL,
    alias_type VARCHAR(32) NOT NULL DEFAULT 'alias',
    confidence DOUBLE NOT NULL DEFAULT 1.0,
    source VARCHAR(64) NOT NULL DEFAULT 'system',
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_evorag_entity_alias_scope_alias_status (
        scope_hash, normalized_alias, status
    ),
    INDEX idx_evorag_entity_alias_entity (entity_id),
    INDEX idx_evorag_entity_alias_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_entity_alias_status (status),
    CONSTRAINT fk_evorag_entity_alias_entity
        FOREIGN KEY (entity_id) REFERENCES evorag_entities(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_ingest_jobs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    source_note_id VARCHAR(128) NOT NULL DEFAULT '',
    task_name VARCHAR(255) NOT NULL DEFAULT '',
    input_fingerprint CHAR(64) NOT NULL,
    input_text MEDIUMTEXT NOT NULL,
    preprocess_json JSON NOT NULL,
    block_count INT NOT NULL DEFAULT 0,
    entity_count INT NOT NULL DEFAULT 0,
    queued_count INT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_evorag_ingest_jobs_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_ingest_jobs_status (status),
    INDEX idx_evorag_ingest_jobs_fingerprint (input_fingerprint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_incoming_entities (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    job_id BIGINT NOT NULL,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    dedupe_key CHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(64) NOT NULL DEFAULT 'concept',
    aliases_json JSON NOT NULL,
    identity_description TEXT NOT NULL,
    description_for_match TEXT NOT NULL,
    attributes_json JSON NOT NULL,
    source_blocks_json JSON NOT NULL,
    source_count INT NOT NULL DEFAULT 1,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    attempt_count INT NOT NULL DEFAULT 0,
    matched_entity_id BIGINT NULL,
    decision VARCHAR(32) NOT NULL DEFAULT '',
    decision_score DOUBLE NOT NULL DEFAULT 0,
    decision_reason TEXT NOT NULL,
    last_error TEXT NOT NULL,
    locked_by VARCHAR(128) NOT NULL DEFAULT '',
    locked_until TIMESTAMP NULL DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_evorag_incoming_entity_dedupe (dedupe_key),
    INDEX idx_evorag_incoming_entities_job (job_id),
    INDEX idx_evorag_incoming_entities_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_incoming_entities_status (status),
    INDEX idx_evorag_incoming_entities_name (normalized_name),
    CONSTRAINT fk_evorag_incoming_entities_job
        FOREIGN KEY (job_id) REFERENCES evorag_ingest_jobs(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_evorag_incoming_entities_matched_entity
        FOREIGN KEY (matched_entity_id) REFERENCES evorag_entities(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_review_tasks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    incoming_entity_id BIGINT NOT NULL,
    job_id BIGINT NOT NULL,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    incoming_snapshot_json JSON NOT NULL,
    candidates_json JSON NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    decision VARCHAR(32) NOT NULL DEFAULT '',
    decided_entity_id BIGINT NULL,
    decided_by VARCHAR(128) NOT NULL DEFAULT '',
    reason TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_evorag_review_incoming_status (incoming_entity_id, status),
    INDEX idx_evorag_review_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_review_status (status),
    INDEX idx_evorag_review_job (job_id),
    CONSTRAINT fk_evorag_review_incoming
        FOREIGN KEY (incoming_entity_id) REFERENCES evorag_incoming_entities(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_evorag_review_decided_entity
        FOREIGN KEY (decided_entity_id) REFERENCES evorag_entities(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_resolution_rejections (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    scope_hash CHAR(40) NOT NULL,
    incoming_normalized_name VARCHAR(255) NOT NULL,
    incoming_type VARCHAR(64) NOT NULL DEFAULT '',
    candidate_entity_id BIGINT NOT NULL,
    reason TEXT NOT NULL,
    decided_by VARCHAR(128) NOT NULL DEFAULT '',
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_evorag_rejection_pair (
        scope_hash, incoming_normalized_name, incoming_type, candidate_entity_id, status
    ),
    INDEX idx_evorag_rejection_scope (workspace_id, project_id, collection_id, domain),
    INDEX idx_evorag_rejection_candidate (candidate_entity_id),
    CONSTRAINT fk_evorag_rejection_candidate
        FOREIGN KEY (candidate_entity_id) REFERENCES evorag_entities(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS evorag_entity_resolution_relation_memory (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    workspace_id VARCHAR(128) NOT NULL DEFAULT 'local',
    project_id VARCHAR(128) NOT NULL DEFAULT 'evorag',
    collection_id VARCHAR(128) NOT NULL DEFAULT 'default',
    domain VARCHAR(128) NOT NULL DEFAULT 'general',
    scope_hash CHAR(40) NOT NULL,
    left_entity_id BIGINT NULL,
    left_name VARCHAR(255) NOT NULL DEFAULT '',
    left_normalized_name VARCHAR(255) NOT NULL,
    left_type VARCHAR(64) NOT NULL DEFAULT '',
    right_entity_id BIGINT NULL,
    right_name VARCHAR(255) NOT NULL DEFAULT '',
    right_normalized_name VARCHAR(255) NOT NULL,
    right_type VARCHAR(64) NOT NULL DEFAULT '',
    decision VARCHAR(32) NOT NULL,
    relation_type VARCHAR(64) NOT NULL DEFAULT '',
    confidence DOUBLE NOT NULL DEFAULT 1.0,
    hit_count BIGINT NOT NULL DEFAULT 0,
    source VARCHAR(64) NOT NULL DEFAULT 'manual',
    reason TEXT NOT NULL,
    evidence_json JSON NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_evorag_relation_memory_pair (
        scope_hash, left_normalized_name, left_type, right_normalized_name, right_type, status
    ),
    INDEX idx_evorag_relation_memory_left (scope_hash, left_normalized_name, left_type, status),
    INDEX idx_evorag_relation_memory_right (scope_hash, right_normalized_name, right_type, status),
    INDEX idx_evorag_relation_memory_decision (decision, status),
    INDEX idx_evorag_relation_memory_hits (hit_count, updated_at),
    CONSTRAINT fk_evorag_relation_memory_left_entity
        FOREIGN KEY (left_entity_id) REFERENCES evorag_entities(id)
        ON DELETE SET NULL,
    CONSTRAINT fk_evorag_relation_memory_right_entity
        FOREIGN KEY (right_entity_id) REFERENCES evorag_entities(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
