import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

const links = [
  { path: '/',         label: 'Home' },
  { path: '/teams',    label: 'Teams' },
  { path: '/circuits', label: 'Circuits' },
  { path: '/mpg',      label: 'Predict MPG' },
  { path: '/laptime',  label: 'Predict Lap' },
  { path: '/winprob',  label: 'Win Probability' },
]

export default function Navbar() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
      background: 'rgba(10,10,10,0.95)',
      backdropFilter: 'blur(10px)',
      borderBottom: '1px solid #222',
      padding: '0 2rem',
      display: 'flex', alignItems: 'center',
      justifyContent: 'space-between', height: '60px'
    }}>
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ color: '#e10600', fontSize: '1.4rem', fontWeight: 700 }}>
          AERO
        </span>
        <span style={{ fontSize: '1.4rem', fontWeight: 300 }}>SPEED</span>
      </Link>

      <div style={{ display: 'flex', gap: '0.25rem' }}>
        {links.map(link => (
          <Link key={link.path} to={link.path}>
            <motion.div
              whileHover={{ scale: 1.05 }}
              style={{
                padding: '0.4rem 0.9rem',
                borderRadius: '6px',
                fontSize: '0.85rem',
                color: pathname === link.path ? '#fff' : '#888',
                background: pathname === link.path ? '#e10600' : 'transparent',
                transition: 'all 0.2s',
                cursor: 'pointer'
              }}
            >
              {link.label}
            </motion.div>
          </Link>
        ))}
      </div>
    </nav>
  )
}
