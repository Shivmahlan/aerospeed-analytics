import { useEffect, useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import axios from 'axios'

const API = 'http://localhost:8000'

const TEAM_COLORS = {
  'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D',
  'Mercedes': '#27F4D2',        'McLaren': '#FF8000',
  'Aston Martin': '#229971',    'Alpine': '#FF87BC',
  'Williams': '#64C4FF',        'AlphaTauri': '#6692FF',
  'Alfa Romeo': '#C92D4B',      'Haas F1 Team': '#B6BABD',
  'RB': '#6692FF',              'Kick Sauber': '#52E252',
}

function SpeedLines() {
  const lines = Array.from({ length: 30 }, (_, i) => i)
  return (
    <div style={{
      position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none'
    }}>
      {lines.map(i => (
        <motion.div key={i}
          initial={{ x: '-100%', opacity: 0 }}
          animate={{ x: '200%', opacity: [0, 0.4, 0] }}
          transition={{
            duration: Math.random() * 2 + 1,
            delay: Math.random() * 4,
            repeat: Infinity,
            repeatDelay: Math.random() * 3
          }}
          style={{
            position: 'absolute',
            top: `${Math.random() * 100}%`,
            height: `${Math.random() * 1.5 + 0.5}px`,
            width: `${Math.random() * 200 + 100}px`,
            background: `linear-gradient(90deg, transparent, #e10600, transparent)`,
            borderRadius: '2px'
          }}
        />
      ))}
    </div>
  )
}

function StatCard({ label, value, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.6 }}
      style={{
        background: '#111',
        border: '1px solid #222',
        borderTop: '3px solid #e10600',
        borderRadius: '8px',
        padding: '1.5rem',
        textAlign: 'center',
        flex: 1
      }}
    >
      <div style={{ fontSize: '2.2rem', fontWeight: 700, color: '#e10600' }}>
        {value}
      </div>
      <div style={{ color: '#888', fontSize: '0.85rem', marginTop: '0.25rem' }}>
        {label}
      </div>
    </motion.div>
  )
}

export default function Home() {
  const [overview, setOverview] = useState(null)
  const [teams, setTeams]       = useState([])

  useEffect(() => {
    axios.get(`${API}/data/overview`).then(r => setOverview(r.data))
    axios.get(`${API}/data/teams?year=2024`).then(r => setTeams(r.data))
  }, [])

  return (
    <div style={{ paddingTop: '60px' }}>

      {/* ── Hero ── */}
      <div style={{
        position: 'relative', minHeight: '100vh',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        overflow: 'hidden', padding: '2rem'
      }}>
        <SpeedLines />

        {/* Grid background */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: `
            linear-gradient(rgba(225,6,0,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(225,6,0,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px'
        }} />

        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          style={{ textAlign: 'center', zIndex: 1, maxWidth: '800px' }}
        >
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: '80px' }}
            transition={{ duration: 0.8, delay: 0.3 }}
            style={{
              height: '3px', background: '#e10600',
              margin: '0 auto 1.5rem', borderRadius: '2px'
            }}
          />

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            style={{
              fontSize: 'clamp(2.5rem, 6vw, 5rem)',
              fontWeight: 800, lineHeight: 1.1,
              marginBottom: '1rem'
            }}
          >
            <span style={{ color: '#e10600' }}>AERO</span>SPEED
            <br />
            <span style={{ fontWeight: 300, fontSize: '60%', color: '#aaa' }}>
              ANALYTICS
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            style={{
              color: '#888', fontSize: '1.1rem',
              maxWidth: '500px', margin: '0 auto 2.5rem', lineHeight: 1.7
            }}
          >
            F1 telemetry meets road car aerodynamics.
            3 seasons · 25 circuits · ML-powered predictions.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1 }}
            style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}
          >
            <Link to="/teams">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                style={{
                  background: '#e10600', color: '#fff',
                  border: 'none', padding: '0.8rem 2rem',
                  borderRadius: '6px', fontSize: '1rem',
                  fontWeight: 600, cursor: 'pointer'
                }}
              >
                Explore Teams
              </motion.button>
            </Link>
            <Link to="/winprob">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                style={{
                  background: 'transparent', color: '#fff',
                  border: '1px solid #444', padding: '0.8rem 2rem',
                  borderRadius: '6px', fontSize: '1rem',
                  cursor: 'pointer'
                }}
              >
                Win Probability
              </motion.button>
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          style={{
            position: 'absolute', bottom: '2rem',
            color: '#444', fontSize: '0.8rem',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', gap: '0.5rem'
          }}
        >
          <span>scroll</span>
          <div style={{ width: '1px', height: '40px', background: '#333' }} />
        </motion.div>
      </div>

      {/* ── Stats row ── */}
      {overview && (
        <div style={{ padding: '4rem 2rem', maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <StatCard label="Total Laps"  value={overview.total_laps.toLocaleString()} delay={0.1} />
            <StatCard label="Circuits"    value={overview.circuits}  delay={0.2} />
            <StatCard label="F1 Teams"    value={overview.teams}     delay={0.3} />
            <StatCard label="Road Cars"   value={overview.road_cars.toLocaleString()} delay={0.4} />
          </div>
        </div>
      )}

      {/* ── Team cards ── */}
      <div style={{ padding: '2rem 2rem 6rem', maxWidth: '1200px', margin: '0 auto' }}>
        <motion.h2
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          style={{ fontSize: '1.8rem', marginBottom: '2rem' }}
        >
          2024 Team Performance
        </motion.h2>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '1rem'
        }}>
          {teams.map((team, i) => {
            const color = TEAM_COLORS[team.team] || '#e10600'
            return (
              <motion.div
                key={team.team}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                whileHover={{ y: -4, scale: 1.02 }}
                transition={{ delay: i * 0.05 }}
                style={{
                  background: '#111',
                  border: '1px solid #222',
                  borderLeft: `4px solid ${color}`,
                  borderRadius: '8px',
                  padding: '1.5rem',
                  cursor: 'pointer'
                }}
              >
                <div style={{
                  fontSize: '0.7rem', color, textTransform: 'uppercase',
                  letterSpacing: '2px', marginBottom: '0.5rem'
                }}>
                  #{i + 1} — 2024
                </div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem' }}>
                  {team.team}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ color: '#888', fontSize: '0.75rem' }}>Median lap</div>
                    <div style={{ color, fontWeight: 600 }}>{team.median_lap}s</div>
                  </div>
                  <div>
                    <div style={{ color: '#888', fontSize: '0.75rem' }}>Top speed</div>
                    <div style={{ fontWeight: 600 }}>{team.top_speed} km/h</div>
                  </div>
                  <div>
                    <div style={{ color: '#888', fontSize: '0.75rem' }}>Avg tyre</div>
                    <div style={{ fontWeight: 600 }}>{team.avg_tyre_life} laps</div>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
