import { useState } from 'react'
import Nav from './Nav'
import Hero from './Hero'
import { BrowserRouter as Router,Routes,Route } from 'react-router-dom'
import Talk from './Talk'
import Login from './Login'
import Signup  from './Pages/Signup'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div className='bg-blue-100 h-vh'>

        <Router>
          <Routes>
            <Route path='/' element={<Hero/>} />
            <Route path='/talk' element={<Talk/>} />
            <Route path='/login' element={<Login/>} />
            <Route path='/signup' element={<Talk/>} />

          </Routes>
        </Router>
        
      </div>
      </>
  )
}

export default App
