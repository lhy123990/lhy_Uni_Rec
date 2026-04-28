2026-04-28 11:22:28.825 2026-04-28 11:22:28,825 INFO Online memory benchmark started: data_dir=/data_ams/academic_training_data schema_path=/data_ams/academic_training_data/schema.json device=cuda:0 batch_sizes=[512] memory_budget_gib=19.00 seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:512

2026-04-28 11:22:28.826 2026-04-28 11:22:28,825 INFO Loading data for batch_size=512

2026-04-28 11:22:29.333 2026-04-28 11:22:29,332 INFO PCVRParquetDataset: 1010000 rows from 1000 file(s), batch_size=512, buffer_batches=0, shuffle=False

2026-04-28 11:22:29.556 2026-04-28 11:22:29,555 INFO Running baseline memory probe for batch_size=512

2026-04-28 11:22:29.562 2026-04-28 11:22:29,562 INFO RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1

2026-04-28 11:22:29.581 2026-04-28 11:22:29,580 INFO RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0

2026-04-28 11:22:31.579 2026-04-28 11:22:31,579 INFO emb_skip_threshold=1000000: seq_b skipped 1/13 features

2026-04-28 11:22:31.579 2026-04-28 11:22:31,579 INFO emb_skip_threshold=1000000: seq_c skipped 3/11 features

2026-04-28 11:22:35.888 2026-04-28 11:22:35,887 INFO Activation checkpointing mode=none units=[]

2026-04-28 11:22:42.189 2026-04-28 11:22:42,188 INFO Running checkpoint equivalence check

2026-04-28 11:22:42.196 2026-04-28 11:22:42,196 INFO RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1

2026-04-28 11:22:42.215 2026-04-28 11:22:42,214 INFO RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0

2026-04-28 11:22:44.207 2026-04-28 11:22:44,207 INFO emb_skip_threshold=1000000: seq_b skipped 1/13 features

2026-04-28 11:22:44.207 2026-04-28 11:22:44,207 INFO emb_skip_threshold=1000000: seq_c skipped 3/11 features

2026-04-28 11:22:47.898 2026-04-28 11:22:47,898 INFO RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1

2026-04-28 11:22:47.916 2026-04-28 11:22:47,916 INFO RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0

2026-04-28 11:22:49.923 2026-04-28 11:22:49,923 INFO emb_skip_threshold=1000000: seq_b skipped 1/13 features

2026-04-28 11:22:49.923 2026-04-28 11:22:49,923 INFO emb_skip_threshold=1000000: seq_c skipped 3/11 features

2026-04-28 11:22:53.606 2026-04-28 11:22:53,605 INFO Activation checkpointing mode=none units=[]

2026-04-28 11:23:02.263 2026-04-28 11:23:02,262 INFO Checkpoint recommendation: --activation_checkpoint_mode none

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Recommendation reason: Batch size 512 fits the memory budget; checkpointing is optional unless you raise batch size further.

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Memory budget: 19.00 GiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Top checkpoint candidates:

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO blocks.0.seq_encoders.seq_c saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO blocks.0.seq_encoders.seq_d saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO blocks.1.seq_encoders.seq_c saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO blocks.1.seq_encoders.seq_d saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Device: cuda:0 (NVIDIA H20), total=95.00 GiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Data: /data_ams/academic_training_data

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Schema: /data_ams/academic_training_data/schema.json

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO batch=512 baseline peak_alloc=13.657 GiB peak_reserved=15.211 GiB gpu=2.0131s wall=2.0132s loss=0.313812

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO batch=512 unit=blocks.0.seq_encoders.seq_c saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO batch=512 unit=blocks.0.seq_encoders.seq_d saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO batch=512 unit=blocks.1.seq_encoders.seq_c saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO batch=512 unit=blocks.1.seq_encoders.seq_d saved_activation=3672.20 MiB

2026-04-28 11:23:02.263 2026-04-28 11:23:02,263 INFO Checkpoint equivalence units=<none> logits_max_abs_diff=0.0000000000e+00 pass_logits=True grad_max_abs_diff=2.9802322388e-08 pass_grads=True compared_dense_grad_tensors=478