import React from 'react'
import { Link } from 'react-router-dom'
import Nav from './Nav'

const Hero = () => {
  return (
    <div >
        <Nav/>
        <div className='hidden md:flex md:justify-between h-screen pt-20 '>
            <div className='px-28 py-44'>
                <h1 className='text-4xl pb-3 font-bold italic text-blue-600'>
                "Inspiring pathways to success.
                </h1>
                <h1 className='text-4xl pb-5 font-bold italic text-blue-600'>
                    Your goals, our expertise"
                </h1>
                <div className='flex flex-col items-center'>
                <h1 className='text-5xl font-extrabold py-4'>
                Try Level to do your Level Best!
                </h1>
                <Link to="/signup">
                <button className='transition ease-in-out duration-300 my-8 py-5 text-white px-10  hover:ring-4 hover:ring-blue-800 hover:bg-blue-950 hover:scale-105 rounded-lg text-2xl font-semibold bg-blue-500'>Get Started</button>
                </Link>
            </div>
            </div>
            <div className='w-[48rem] pr-28'>
                <img className='transition ease-in-out duration-700 hover:scale-110 py-12 ' src="https://appinventiv.com/wp-content/uploads/sites/1/2021/08/chatbot-development-with-deep-nlp.png" alt="webLanding" />
            </div>
        </div>

        <div className='md:hidden pt-8 h-screen'>
            <div >
                <img className='transition ease-in-out duration-700 p-8 hover:scale-110' src="https://blog.happyfox.com/wp-content/uploads/2020/10/How-to-build-chatbots-with-personality_.png" alt="phonelanding" />
            </div>
            <div className='w-[90%] mx-auto flex flex-col items-center font-semibold'>
                <p className='italic'>
                Inspiring pathways to success. Your goals, our expertise
                </p>
                <h1 className='py-2 text-xl font-bold'>Try Level to do your Level Best!</h1>
                <Link to="/signup">
                <button className='transition ease-in-out duration-300 my-3 py-3 text-white px-5  hover:ring hover:bg-blue-950 hover:scale-105 rounded-lg text-lg font-bold bg-blue-500'
                >Get Started</button>
                </Link>
            </div>
        </div>
    </div>
  )
}

export default Hero