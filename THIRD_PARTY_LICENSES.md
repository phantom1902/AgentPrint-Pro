# Third-party implementation licenses

This inventory describes the bundled copies used by the suite. Implementation provenance is recorded in [SOURCES.md](SOURCES.md); individual vendored file hashes are in [VENDORED_FILES.json](VENDORED_FILES.json). A paper citation or public source repository is not, by itself, a redistribution license.

| Baseline | Bundled implementation | License evidence and status |
| --- | --- | --- |
| FlowLens | `Benchmark/2021-NDSS-FlowLens/` | No standalone license found in the bundled copy; upstream permission unresolved. |
| YaTC | `Benchmark/2023-AAAI-YaTC/` | No standalone license found in the bundled copy; upstream permission unresolved. File-level upstream notices still apply. |
| NetMamba | `Benchmark/2024-ICNP-NetMamba/` | No top-level license found in the bundled copy; permission for NetMamba itself unresolved. Its nested Mamba dependency is separately licensed below. |
| TFusion | `Benchmark/2025-CCS-TFusion/` | GNU GPL version 3 text in [upstream LICENSE](Benchmark/2025-CCS-TFusion/LICENSE); retain any file-specific version notices. |
| TrafficFormer | `Benchmark/2025-SP-TrafficFormer/` | [MIT](Benchmark/2025-SP-TrafficFormer/LICENSE); additional file-level notices remain in effect. |
| DF | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| AWF | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| TikTok | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| VarCNN | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| NetCLR | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| TMWF | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| ARES | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |
| BAPM | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/` | WFlib implementation, [MIT](Benchmark/2024_CCS_WFlib_Early_Stage_WF/LICENSE). |

The nested `Benchmark/2024-ICNP-NetMamba/mamba-1p1p1/` dependency has an [Apache License 2.0](Benchmark/2024-ICNP-NetMamba/mamba-1p1p1/LICENSE) and an [AUTHORS](Benchmark/2024-ICNP-NetMamba/mamba-1p1p1/AUTHORS) file. These do not establish a license for the surrounding NetMamba implementation. Embedded code can carry additional upstream notices; preserve those as well.

The eight WF entries refer to the WFlib implementations actually bundled here, not eight independently downloaded official implementations. This inventory does not replace file-level attribution or establish that every transitive contribution has been independently cleared.

Before a public redistribution of the complete bundle, resolve the missing license grants for FlowLens, YaTC, and NetMamba, or distribute those components separately through an authorized acquisition process. The root GPL grant does not supply that missing permission. Third-party copyright holders and contacts are retained as required attribution; they are not identified as ANT benchmark authors.

Collection requirements and task references do not bundle or relicense the corresponding external packages, datasets, websites, agent services, or weights.
