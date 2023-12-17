import React from 'react'
import Logo from './images/Logo.png'
import { Link } from 'react-router-dom'

const Nav = () => {
  return (
    <div className='bg-blue-500 flex justify-between h-16 md:h-[70px]'>
      <Link to="/">
        <div className=' flex py-4 md:py-3 sm:pl-2 pl-1 md:pl-5 md:ml-20  '>
            <img className='w-8 h-8 md:w-10 md:h-10  rounded-full' src="https://st2.depositphotos.com/33928496/48382/v/450/depositphotos_483828962-stock-illustration-logo-letters-monogram-prisma-shape.jpg" alt="Logo" />
            <h1 className='px-2 pt-1 sm:pt-0 text-lg font-medium sm:px-3 md:px-4 md:pt-[2px] sm:text-2xl md:text-3xl sm:font-semibold md:font-bold text-black'>LevelAi</h1>
        </div>
        </Link>
        <div className=' flex justify-end gap-3 sm:gap-6 md:gap-14'>
            {/* <Link to="/login">
            <button className='my-3 text-white px-4 py-2 hover:ring hover:ring-white md:px-6 rounded-lg text-md md:text-lg font-bold bg-blue-950'>Login</button>
            </Link> */}
            <Link to="/talk">
            <button className='my-3 text-white px-4 py-2 hover:ring hover:ring-white md:px-10  rounded-lg mr-3 sm:mr-5 text-md md:mr-32 md:text-lg font-bold bg-blue-950'>Try Level</button>
             </Link>
        </div>
    </div>
  )
}

export default Nav


