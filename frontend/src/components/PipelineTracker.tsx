import React from 'react'

interface PipelineTrackerProps {
  activeStage?: number
  stages?: StageDef[]
  title?: string
  subtitle?: string
}

interface StageDef {
  id: number
  label: string
  locked?: boolean
  state?: 'completed' | 'active' | 'locked' | 'idle' | 'skipped'
}

const DEFAULT_STAGES: StageDef[] = [
  { id: 1, label: 'Encode' },
  { id: 2, label: 'Lift 3D' },
  { id: 3, label: 'Refine' },
  { id: 4, label: 'Compare' },
  { id: 5, label: 'Classify' },
]

const CheckIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

const LockIcon = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
)

const ActiveDot = () => (
  <svg width="6" height="6" viewBox="0 0 6 6">
    <circle cx="3" cy="3" r="2" fill="currentColor" />
  </svg>
)

function getStageState(stageId: number, activeStage: number, stages: StageDef[]): 'completed' | 'active' | 'locked' | 'idle' | 'skipped' {
  const stage = stages.find(s => s.id === stageId)
  if (stage?.state) return stage.state
  if (stage?.locked && stageId > activeStage) return 'locked'
  if (stageId < activeStage) return 'completed'
  if (stageId === activeStage) return 'active'
  return 'idle'
}

export default function PipelineTracker({ activeStage = 0, stages = DEFAULT_STAGES, title, subtitle }: PipelineTrackerProps) {
  const elements: React.JSX.Element[] = []
  stages.forEach((stage, i) => {
    const state = getStageState(stage.id, activeStage, stages)
    elements.push(
      <div key={`s${stage.id}`} className="pipeline-stage">
        <div className={`pipeline-circle ${state}`}>
          {state === 'completed' && <CheckIcon />}
          {state === 'active' && <ActiveDot />}
          {state === 'locked' && <LockIcon />}
          {state === 'skipped' && (
            <span style={{ width: 10, height: 1, background: 'currentColor', opacity: 0.45, display: 'block' }} />
          )}
          {state === 'idle' && (
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--text-muted)', opacity: 0.3, display: 'block' }} />
          )}
        </div>
        <span className={`pipeline-label ${state}`}>{stage.label}</span>
      </div>
    )
    if (i < stages.length - 1) {
      const lineCompleted = ['completed', 'active', 'skipped'].includes(state)
      elements.push(
        <div key={`l${stage.id}`} className={`pipeline-line ${lineCompleted ? 'completed' : ''}`} />
      )
    }
  })

  return (
    <div className="pipeline-bar-wrap">
      {(title || subtitle) && (
        <div className="pipeline-meta">
          {title && <div className="pipeline-title">{title}</div>}
          {subtitle && <div className="pipeline-subtitle">{subtitle}</div>}
        </div>
      )}
      <div className="pipeline-bar">
        <div className="pipeline-stages">
          {elements}
        </div>
      </div>
    </div>
  )
}

PipelineTracker.displayName = 'PipelineTracker'
