import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Teams from './pages/Teams'
import Circuits from './pages/Circuits'
import PredictMPG from './pages/PredictMPG'
import PredictLap from './pages/PredictLap'
import WinProb from './pages/WinProb'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"          element={<Home />} />
        <Route path="/teams"     element={<Teams />} />
        <Route path="/circuits"  element={<Circuits />} />
        <Route path="/mpg"       element={<PredictMPG />} />
        <Route path="/laptime"   element={<PredictLap />} />
        <Route path="/winprob"   element={<WinProb />} />
      </Routes>
    </BrowserRouter>
  )
}
