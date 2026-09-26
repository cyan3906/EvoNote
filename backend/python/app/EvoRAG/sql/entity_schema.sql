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
