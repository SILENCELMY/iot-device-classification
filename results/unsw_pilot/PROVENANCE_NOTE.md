# 版本控制副本说明

`DATASET_README_SOURCE.md`（许可与来源记录）与 `device_mac_map.csv`（MAC→设备映射）
的权威原件在服务器 `dataset/unsw/`（该目录按 §19.7 不入库）；此处为入库副本，与原件逐字节一致。
分析脚本已移至 `code/scripts/analysis/unsw_pilot/`。

## 下载链路取证（2026-08-29，回应"直连为何有 ~10MB/s"的质询）

- `iotanalytics.unsw.edu.au` 为 CNAME 至 `d3mvmotae8r6uu.cloudfront.net`：**UNSW 官方将数据集
  托管在 Amazon CloudFront CDN**，下载实际连接的是就近边缘节点（18.65.14.x，实测 RTT ≈ 89ms，
  东亚边缘量级；悉尼直连通常 150–300ms）——这是 8.5–9.8 MB/s 吞吐的全部解释。
- **服务器侧零代理**（逐层取证）：env 无 proxy 变量；~/.bashrc、~/.profile、/etc/environment 无
  proxy 行；无 ~/.wgetrc、~/.curlrc；/etc/wgetrc 无有效 proxy 行；git 无 proxy 配置；wget 日志
  显示逐次 "Connecting to iotanalytics.unsw.edu.au|18.65.14.29|:443... connected"（无代理主机）；
  下载脚本另加显式 --no-proxy。
- CDN 是**发布方自己的分发基础设施**（对所有访问者一视同仁），不是本机配置或使用的代理；
  本机自始至终直连官方域名。
