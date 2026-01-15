================================================================================
CR_Score PERMISSIONS MATRIX
================================================================================

Version: 1.0
Date: 2026-01-15
Purpose: Define role-based access control (RBAC) for all CR_Score operations

================================================================================
1. USER ROLES & DEFINITIONS
================================================================================

VIEWER
------
Purpose: Read-only access to completed runs and reports
Permissions: View only, no modifications
Typical Users: Executives, Risk managers, Compliance officers

ANALYST
-------
Purpose: Create and run analyses, basic scorecards
Permissions: Create configs, run EDA, view binning, create runs
Typical Users: Junior data scientists, credit analysts

MODELER
-------
Purpose: Full modeling capabilities, manual overrides
Permissions: All analyst permissions + manual overrides, feature engineering
Typical Users: Senior data scientists, quant analysts

VALIDATOR
---------
Purpose: Reproducibility testing, run locking, approval
Permissions: Compare runs, lock artifacts, approve for deployment
Typical Users: Model validation team, compliance

ADMIN
-----
Purpose: System administration, user management
Permissions: All operations, user role assignment, system configuration
Typical Users: Platform administrators

================================================================================
2. PERMISSION MATRIX BY OPERATION
================================================================================

CONFIG MANAGEMENT
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View config                      ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Create new config                ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Edit config                      ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Delete config                    ║   ✗   ║   ✗    ║    ✓    ║     ✗      ║   ✓   ║
║ Share config (change visibility) ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

RUN EXECUTION
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View completed runs              ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Create and run scorecard         ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Cancel running job               ║   ✗   ║   ✓*   ║    ✓    ║     ✗      ║   ✓   ║
║ Delete run artifacts             ║   ✗   ║   ✗    ║    ✓    ║     ✗      ║   ✓   ║
║ Export run for deployment        ║   ✗   ║   ✗    ║    ✓    ║     ✓      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝
 * Analyst can cancel only own runs

BINNING & MANUAL OVERRIDES
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View binning tables              ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Auto-generate bins               ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Manual bin override              ║   ✗   ║   ✗    ║    ✓    ║     ✗      ║   ✓   ║
║ Enforce monotonicity             ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Merge bins                       ║   ✗   ║   ✗    ║    ✓    ║     ✗      ║   ✓   ║
║ View override audit trail        ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

MODEL BUILDING
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View model performance           ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Select features                  ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Train/retrain model              ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Adjust regularization parameters ║   ✗   ║   ✗    ║    ✓    ║     ✗      ║   ✓   ║
║ Compare two models               ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

VALIDATION & APPROVAL
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View reproducibility results     ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Run reproducibility tests        ║   ✗   ║   ✗    ║    ✓    ║     ✓      ║   ✓   ║
║ Compare runs (golden comparison) ║   ✗   ║   ✗    ║    ✓    ║     ✓      ║   ✓   ║
║ Lock run (immutable)             ║   ✗   ║   ✗    ║    ✗    ║     ✓      ║   ✓   ║
║ Approve for deployment           ║   ✗   ║   ✗    ║    ✗    ║     ✓      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

REPORTING & EXPORT
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View reports                     ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Generate report                  ║   ✗   ║   ✓    ║    ✓    ║     ✗      ║   ✓   ║
║ Export scoring spec (JSON)       ║   ✗   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Export model card                ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Download artifacts               ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

MONITORING & MAINTENANCE
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ View monitoring dashboards       ║   ✓   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Configure monitoring             ║   ✗   ║   ✗    ║    ✗    ║     ✗      ║   ✓   ║
║ View audit logs                  ║   ✗   ║   ✓    ║    ✓    ║     ✓      ║   ✓   ║
║ Export audit logs                ║   ✗   ║   ✗    ║    ✗    ║     ✓      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

SYSTEM ADMINISTRATION
╔═════════════════════════════════╦═══════╦════════╦═════════╦════════════╦═══════╗
║ Operation                        ║ Viewer║ Analyst║ Modeler ║ Validator  ║ Admin ║
╠═════════════════════════════════╬═══════╬════════╬═════════╬════════════╬═══════╣
║ Manage users                     ║   ✗   ║   ✗    ║    ✗    ║     ✗      ║   ✓   ║
║ Assign roles                     ║   ✗   ║   ✗    ║    ✗    ║     ✗      ║   ✓   ║
║ Configure system settings        ║   ✗   ║   ✗    ║    ✗    ║     ✗      ║   ✓   ║
║ View system health               ║   ✗   ║   ✗    ║    ✗    ║     ✗      ║   ✓   ║
║ Manage storage/backups           ║   ✗   ║   ✗    ║    ✗    ║     ✗      ║   ✓   ║
╚═════════════════════════════════╩═══════╩════════╩═════════╩════════════╩═══════╝

================================================================================
3. MANUAL OVERRIDE REQUIREMENTS
================================================================================

When Override is Required:

    ✓ Manual bin override (change auto-generated bins)
    ✓ Change feature selection
    ✓ Adjust model regularization
    ✓ Modify score scaling parameters
    ✗ NOT required for: config changes, run execution, EDA

Override Approval Workflow:

    1. MODELER initiates override
       - Specifies: resource (e.g., "age" binning)
       - Provides: reason (mandatory, not empty)
       - Documents: before/after values
       
    2. SYSTEM logs:
       - User ID who made override
       - Timestamp
       - Full diff
       - Reason
       - Audit log entry (immutable)
       
    3. VALIDATOR notified:
       - Alert sent to all validators
       - Run marked as "manual_override_pending_review"
       - Cannot be deployed until validator reviews
       
    4. VALIDATOR reviews:
       - Examines override reason and impact
       - Approves or rejects
       - Documents decision

Override Rejection:

    If validator rejects:
    - Override cannot be undone (history immutable)
    - New run must be created without override
    - Reason for rejection logged in audit trail
    - Notification sent to originating modeler

================================================================================
4. AUDIT LOGGING REQUIREMENTS
================================================================================

Audit Log Entry (audit_log.jsonl):

    {
        "timestamp": "2026-01-15T10:45:23Z",
        "audit_id": "audit_20260115_001_12345",
        "correlation_id": "run_20260115_001",
        "user_id": "modeler_user_123",
        "user_role": "MODELER",
        "action": "bin_override",
        "resource_type": "binning_table",
        "resource_id": "run_20260115_001/age",
        "status": "success",
        "before": {
            "bins": [
                {"id": 1, "label": "18-25", "count": 450000},
                {"id": 2, "label": "26-35", "count": 890000}
            ]
        },
        "after": {
            "bins": [
                {"id": 1, "label": "18-18", "count": 50000},
                {"id": 2, "label": "19-25", "count": 400000},
                {"id": 3, "label": "26-35", "count": 890000}
            ]
        },
        "reason": "Business requirement: age < 18 must be separate bin for regulatory compliance",
        "ip_address": "192.168.1.100",
        "source": "api",
        "duration_ms": 2345,
        "result": "completed"
    }

Audit Trail Invariants:

    ☐ All entries append-only (immutable)
    ☐ Timestamps in UTC
    ☐ Correlation ID links to run
    ☐ Before/after values captured for mutations
    ☐ Reason always populated for manual actions
    ☐ User ID never null
    ☐ Never log sensitive data (PII, secrets)

Queryable Audit Log:

    # API endpoints for audit log queries
    GET /audit-logs?user_id=modeler_123&start_date=2026-01-01
    GET /audit-logs?run_id=run_20260115_001
    GET /audit-logs?action=bin_override&resource_type=binning_table
    GET /audit-logs?date_range=2026-01-15&status=success

Audit Log Retention:

    - Retention period: Indefinite (compliance requirement)
    - Archive to cold storage after 1 year
    - Regular backups (daily minimum)
    - Encrypted at rest and in transit

================================================================================
5. VALIDATOR MODE (IMMUTABILITY)
================================================================================

Validator Mode Purpose:

    Once a run is locked by VALIDATOR, it becomes immutable:
    - No modifications to binning
    - No re-runs of same config
    - Ensures reproducibility
    - Prepares for deployment

Locking a Run:

    def lock_run_for_deployment(run_id: str, validator_id: str, reason: str):
        """Lock run to prevent modifications."""
        run_metadata = artifact_index.get(f"{run_id}/metadata")
        run_metadata["locked"] = True
        run_metadata["locked_by"] = validator_id
        run_metadata["locked_at"] = datetime.utcnow().isoformat()
        run_metadata["locked_reason"] = reason
        run_metadata["deployment_approved"] = True
        
        artifact_index.update(run_metadata)
        logger.info(f"Run {run_id} locked for deployment")

Locked Run Guarantees:

    ☐ All artifacts frozen (content hash verified on access)
    ☐ No new overrides allowed
    ☐ Reproducibility test passed before locking
    ☐ Model card finalized
    ☐ Scoring spec exported
    ☐ Monitoring plan in place

Attempting to Modify Locked Run:

    if run_metadata.get("locked"):
        raise PermissionError(
            f"Run {run_id} is locked by {run_metadata['locked_by']} "
            f"for deployment. Cannot modify locked runs."
        )

================================================================================
6. ROLE TRANSITION & DELEGATION
================================================================================

Temporary Role Elevation:

    Analyst → Modeler (temporary, for specific task):
    - Admin assigns temporary role with expiration
    - Default: 24-hour expiration
    - All actions logged with "temporary_role" indicator
    - Expires automatically (no manual revocation needed)

Delegation:

    Modeler delegates binning override to Analyst:
    - Modeler creates override, marks as "delegated_approval"
    - Analyst reviews and approves/rejects
    - Modeler still accountable (logged as originator)
    - Validator gets full context

Role Change Audit:

    {
        "timestamp": "2026-01-15T10:00:00Z",
        "audit_id": "audit_20260115_role_change_001",
        "action": "role_assignment",
        "user_id": "analyst_user_456",
        "old_role": "ANALYST",
        "new_role": "MODELER",
        "assigned_by": "admin_user_789",
        "expiration": "2026-01-16T10:00:00Z",
        "reason": "Temporary elevation for feature selection task"
    }

================================================================================
7. PERMISSION ENFORCEMENT IMPLEMENTATION
================================================================================

Decorator Pattern for Permissions:

    from CR_Score.core.permissions import require_permission
    
    @require_permission("binning_override")
    def manual_bin_override(run_id: str, override_spec: dict):
        """Apply manual bin override."""
        # This function only runs if user has permission
        # Otherwise raises PermissionError with audit log
        ...

Checking Permissions in Code:

    from CR_Score.core.permissions import has_permission, PermissionError
    
    def create_scorecard(config: dict, user_id: str):
        """Create scorecard run."""
        if not has_permission(user_id, "create_run"):
            raise PermissionError(
                f"User {user_id} does not have permission 'create_run'"
            )
        # Continue with logic...

Runtime Permission Checks:

    All permission checks include:
    ☐ User ID extraction from context/token
    ☐ Role lookup from user management system
    ☐ Permission verification against role matrix
    ☐ Audit logging (success and failures)
    ☐ Exception raised if denied

API Authorization:

    @app.post("/runs")
    @auth_required
    @require_permission("create_run")
    def create_run(config: dict, request: Request):
        """Create a new scorecard run."""
        user_id = request.user.id
        # Only reaches here if permission granted
        ...

================================================================================
8. PERMISSION DENIAL SCENARIOS
================================================================================

Scenario 1: Analyst Attempts Manual Override

    POST /runs/{run_id}/bins/age/override
    User: analyst_user_123 (role: ANALYST)
    Reason: "Want to split age bin"
    
    Response:
    403 Forbidden
    {
        "error": "Permission denied",
        "required_permission": "binning_override",
        "user_role": "ANALYST",
        "minimum_role": "MODELER",
        "message": "Only users with MODELER or higher role can perform manual overrides"
    }
    
    Audit Log:
    {
        "timestamp": "...",
        "action": "permission_denied",
        "user_id": "analyst_user_123",
        "attempted_action": "binning_override",
        "status": "denied",
        "reason": "Insufficient role"
    }

Scenario 2: Modeler Attempts System Configuration

    PUT /system/config/spark_partitions
    User: modeler_user_456 (role: MODELER)
    
    Response:
    403 Forbidden
    {
        "error": "Permission denied",
        "required_permission": "system_configuration",
        "minimum_role": "ADMIN"
    }

Scenario 3: Viewer Attempts to Run Scorecard

    POST /runs
    User: viewer_user_789 (role: VIEWER)
    
    Response:
    403 Forbidden
    {
        "error": "Permission denied",
        "required_permission": "create_run",
        "user_role": "VIEWER",
        "message": "Viewers have read-only access"
    }

================================================================================
9. PERMISSION VERIFICATION CHECKLIST
================================================================================

Before Deploy, Verify:

    ☐ All CLI commands have role checks
    ☐ All API endpoints have @require_permission decorators
    ☐ All SDK methods check permissions
    ☐ Manual overrides require reason + logging
    ☐ Validators receive notifications for critical actions
    ☐ Audit logs capture all permission-relevant actions
    ☐ Permission denied errors include helpful guidance
    ☐ Locked runs cannot be modified
    ☐ Temporary roles expire correctly
    ☐ Role changes audit-logged with reason

================================================================================
