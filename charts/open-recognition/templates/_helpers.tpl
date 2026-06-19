{{- define "open-recognition.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "open-recognition.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "open-recognition.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "open-recognition.labels" -}}
helm.sh/chart: {{ include "open-recognition.chart" . }}
{{ include "open-recognition.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "open-recognition.selectorLabels" -}}
app.kubernetes.io/name: {{ include "open-recognition.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "open-recognition.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "open-recognition.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Name of the bundled PostgreSQL workload + service.
*/}}
{{- define "open-recognition.postgresqlFullname" -}}
{{- printf "%s-postgresql" (include "open-recognition.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Whether this chart owns the application Secret holding OPEN_RECOGNITION_DATABASE_URL.
False only when pointing at an external database via a pre-existing Secret.
*/}}
{{- define "open-recognition.createDatabaseSecret" -}}
{{- if and (not .Values.postgresql.enabled) .Values.externalDatabase.existingSecret -}}
false
{{- else -}}
true
{{- end -}}
{{- end -}}

{{/*
Name of the Secret the Deployment reads OPEN_RECOGNITION_DATABASE_URL from.
*/}}
{{- define "open-recognition.databaseSecretName" -}}
{{- if and (not .Values.postgresql.enabled) .Values.externalDatabase.existingSecret -}}
{{- .Values.externalDatabase.existingSecret -}}
{{- else -}}
{{- include "open-recognition.fullname" . -}}
{{- end -}}
{{- end -}}

{{/*
Key within that Secret holding the DSN.
*/}}
{{- define "open-recognition.databaseSecretKey" -}}
{{- if and (not .Values.postgresql.enabled) .Values.externalDatabase.existingSecret -}}
{{- .Values.externalDatabase.existingSecretKey -}}
{{- else -}}
OPEN_RECOGNITION_DATABASE_URL
{{- end -}}
{{- end -}}

{{/*
The database DSN rendered into the app Secret. For the bundled Postgres it is
built from postgresql.auth; otherwise it is the externalDatabase.url (required).
*/}}
{{- define "open-recognition.databaseUrl" -}}
{{- if .Values.postgresql.enabled -}}
{{- printf "postgresql://%s:%s@%s:%v/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "open-recognition.postgresqlFullname" .) .Values.postgresql.service.port .Values.postgresql.auth.database -}}
{{- else -}}
{{- required "externalDatabase.url (or externalDatabase.existingSecret) is required when postgresql.enabled=false" .Values.externalDatabase.url -}}
{{- end -}}
{{- end -}}
