<template>
  <div class="metadata-panel" v-if="metadata">
    <div class="metadata-header">
      <div class="metadata-icon" :style="{ background: iconGradient }">
        <el-icon :size="14"><Coin /></el-icon>
      </div>
      <span class="metadata-title">{{ dataset }}</span>
      <span v-if="metadata.sampled" class="sampled-badge">{{ t('meta.panel.sampled') }}</span>
    </div>

    <div class="metadata-grid">
      <template v-for="key in displayKeys" :key="key">
        <div class="meta-item">
          <span class="meta-label">{{ formatKey(key) }}</span>
          <span class="meta-value">{{ formatValue(metadata[key]) }}</span>
        </div>
      </template>
    </div>

    <template v-for="key in nestedKeys" :key="key">
      <div class="nested-section" v-if="typeof metadata[key] === 'object' && metadata[key] !== null">
        <div class="nested-header">{{ formatKey(key) }}</div>
        <div class="nested-grid">
          <div class="meta-item" v-for="(val, subKey) in (metadata[key] as Record<string, any>)" :key="String(subKey)">
            <span class="meta-label">{{ formatKey(String(subKey)) }}</span>
            <span class="meta-value">{{ formatValue(val) }}</span>
          </div>
        </div>
      </div>
    </template>
  </div>

  <div class="metadata-panel metadata-no-data" v-else-if="status === 'downloaded'">
    <el-icon :size="32"><Coin /></el-icon>
    <span class="no-data-text">{{ t('meta.panel.downloadedNotProcessed') }}</span>
    <router-link v-if="showPreprocessLink" :to="{ name: 'preprocess', query: { dataset } }" class="preprocess-link">
      <el-icon :size="14"><Download /></el-icon>
      {{ t('meta.panel.goProcess') }}
    </router-link>
  </div>

  <div class="metadata-panel metadata-no-data" v-else-if="status === 'empty'">
    <el-icon :size="32"><Coin /></el-icon>
    <span class="no-data-text">{{ t('meta.panel.notDownloaded') }}</span>
  </div>

  <div class="metadata-panel metadata-empty" v-else-if="dataset && loading">
    <div class="empty-text">{{ t('meta.panel.loading') }}</div>
  </div>

  <div class="metadata-panel metadata-placeholder" v-else>
    <el-icon :size="32"><Coin /></el-icon>
    <span>{{ t('meta.panel.selectHint') }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { Coin, Download } from '@element-plus/icons-vue'
import { type DatasetMetadata, type DatasetStatus } from '@/api/datasets'

const HIDDEN_KEYS = new Set([
  'question_data_md5', 'sequence_data_md5',
  'split_question_sequence_data_md5', 'split_skill_sequence_data_md5',
  'windowlate_data_md5', 'data_base_path', 'dataset',
])

const KEY_LABELS: Record<string, string> = {
  num_users: 'meta.num_users',
  num_questions: 'meta.num_questions',
  num_skills: 'meta.num_skills',
  num_assignments: 'meta.num_assignments',
  num_templates: 'meta.num_templates',
  num_split_question_users: 'meta.num_split_question_users',
  num_split_skill_users: 'meta.num_split_skill_users',
  kfold_n_splits: 'meta.kfold_n_splits',
  test_ratio: 'meta.test_ratio',
  min_seq_len: 'meta.min_seq_len',
  skill_split_unit: 'meta.skill_split_unit',
  max_question_seq_len: 'meta.max_question_seq_len',
  max_skill_seq_len: 'meta.max_skill_seq_len',
  max_windowlate_seq_len: 'meta.max_windowlate_seq_len',
  random_seed: 'meta.random_seed',
  sampled: 'meta.sampled',
  sampling_config: 'meta.sampling_config',
  sampling_stats: 'meta.sampling_stats',
  n_samples_requested: 'meta.n_samples_requested',
  n_samples_actual: 'meta.n_samples_actual',
  sampling_ratio: 'meta.sampling_ratio',
  stratify: 'meta.stratify',
  attempts_bins: 'meta.attempts_bins',
  correct_bins: 'meta.correct_bins',
  original_users: 'meta.original_users',
  sampled_users: 'meta.sampled_users',
  original_records: 'meta.original_records',
  sampled_records: 'meta.sampled_records',
  strata_distribution: 'meta.strata_distribution',
}

const props = defineProps<{
  dataset: string
  metadata: DatasetMetadata | null
  status?: DatasetStatus
  loading?: boolean
  iconGradient?: string
  showPreprocessLink?: boolean
}>()

const { t } = useI18n()

const displayKeys = computed(() => {
  if (!props.metadata) return []
  return Object.keys(props.metadata).filter(k => !HIDDEN_KEYS.has(k) && typeof props.metadata![k] !== 'object')
})

const nestedKeys = computed(() => {
  if (!props.metadata) return []
  return Object.keys(props.metadata).filter(k => !HIDDEN_KEYS.has(k) && typeof props.metadata![k] === 'object' && props.metadata![k] !== null)
})

function formatKey(key: string): string {
  const mapped = KEY_LABELS[key]
  return mapped ? t(mapped) : key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatValue(val: any): string {
  if (typeof val === 'boolean') return val ? t('common.yes') : t('common.no')
  if (Array.isArray(val)) return val.map(String).join(', ')
  if (typeof val === 'number') return val.toLocaleString()
  return String(val)
}
</script>

<style scoped>
.metadata-panel {
  width: 340px;
  flex-shrink: 0;
  padding: 18px;
  border-left: 1px solid var(--border-muted);
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
  max-height: 420px;
}

.metadata-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.metadata-icon {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  flex-shrink: 0;
}

.metadata-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.sampled-badge {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--accent-orange) 12%, transparent);
  color: var(--accent-orange);
  border: 1px solid color-mix(in srgb, var(--accent-orange) 20%, transparent);
  margin-left: auto;
}

.metadata-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 16px;
}

.meta-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.meta-label {
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 500;
}

.meta-value {
  font-size: 13px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-weight: 600;
}

.nested-section {
  border-top: 1px solid var(--border-muted);
  padding-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.nested-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent-orange);
}

.nested-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 16px;
}

.metadata-empty {
  align-items: center;
  justify-content: center;
}

.empty-text {
  color: var(--text-tertiary);
  font-size: 13px;
}

.metadata-no-data {
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.no-data-text {
  color: var(--text-tertiary);
  font-size: 13px;
}

.preprocess-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  color: var(--accent-blue);
  background: rgba(9, 105, 218, 0.08);
  border: 1px solid rgba(9, 105, 218, 0.2);
  transition: all 0.15s ease;
}

.preprocess-link:hover {
  background: rgba(9, 105, 218, 0.14);
  border-color: var(--accent-blue);
}

.metadata-placeholder {
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.metadata-placeholder span {
  color: var(--text-tertiary);
  font-size: 13px;
}
</style>
