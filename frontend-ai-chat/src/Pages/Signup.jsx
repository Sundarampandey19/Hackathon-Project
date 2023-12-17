import React, { useState } from 'react'
import Nav from '../Nav'

const Signup = () => {
    const [Username,setUsername] = useState('') 
    const [Password,setPassword] = useState('') 

    function SubmitAction(){

        function callback2(data){
            console.log(data)
            localStorage.setItem("token",data.token)
        }

        function callback1(res){
            res.json().then(callback2)
        }
        fetch("http://localhost:3000/signup",{
            method:"POST",
            body:{  
                username:Username,
                password:Password,

            },
            headers:{
                    "Content-type":"application/json",
                    "authentication":`Bearer ${localStorage.getItem("token")}`
                    
            }.then(callback1)
        })
    }




    return (
        <>
        <div className='h-screen'>
        <Nav/>
        <div className=' flex flex-col items-center py-32'>
          
        <div className='bg-blue-300 px-14 pt-14  py-10 rounded-xl flex flex-col items-center'>
        <h1 className='px-5 pt-1 pb-2 shadow-lg shadow-blue-400 mb-4 rounded-md text-3xl font-bold bg-blue-500'>Sign Up!</h1>
        <br />
        <input className='rounded-md py-1 px-1 focus:outline-none' type="text" placeholder='Username' onChange={(e)=>{setUsername(e.target.value)}}/>
        <br />
        <input className='rounded-md py-1 px-1 focus:outline-none' type="text" placeholder='Password' onChange={(e)=>{setPassword(e.target.value)}} />
        <br />
        <button className="my-3 text-white px-4 py-3 hover:shadow-xl hover:shadow-blue-500 hover:ring hover:ring-white md:px-8 rounded-lg text-md md:text-xl font-bold bg-blue-950" onClick={()=>{
            {SubmitAction()}
        }}>Signup</button>
        </div>
        </div>
        </div>
        </>
      )
}

export default Signup