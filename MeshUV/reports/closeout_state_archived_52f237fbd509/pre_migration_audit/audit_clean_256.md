# Clean 256 审计

- accepted 149; yield {'ACCEPTED': 213, 'COVERAGE_REJECTED': 65, 'TIMEOUT': 614, 'PARTUV_FAILED': 1, 'TILED_UV_UNSUPPORTED': 306, 'PRECHECK_REJECTED': 1}
- charts: 总 90762, P50 160, P90 1635, P95 2726, P99 7167, max 8012
- factor≠1: 5; 纯色无UV: 2; 有纹理无UV: 0; UV 越界: 3; 整图平移: 3; 跨 tile: 0
- 缺 v2 面积字段(v1 构建): 0
- label drift(clean 重算 vs 已存): max 0.62230, >1e-4 共 139
- 需重建 UID(跨tile/有纹理无UV/v1字段/错误/schema): 64
- adapter 分布: {'canonicalizer_rgb_v2': 149}
- audit_hash: 52f237fbd509bd5b  commit: 28209d2bb9bb
