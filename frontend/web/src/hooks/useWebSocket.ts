import { useEffect, useRef, useState, useCallback } from 'react';
import type { Job } from '../types';

export function useJobWebSocket(jobId: string | null) {
  const [jobState, setJobState] = useState<Job | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/jobs/${jobId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as Job;
      setJobState(data);
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [jobId]);

  return jobState;
}

export function useJobPolling(jobId: string | null, intervalMs = 2000) {
  const [jobState, setJobState] = useState<Job | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const data = await res.json();
        setJobState(data);
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          clearInterval(timer);
        }
      } catch {
        // ignore
      }
    };

    poll();
    const timer = setInterval(poll, intervalMs);
    return () => clearInterval(timer);
  }, [jobId, intervalMs]);

  return jobState;
}
