// server.js
const express = require('express');
const dotenv = require('dotenv');
const connectDB = require('./config/Db');
const cors = require('cors'); 

dotenv.config();

const app = express();

connectDB();

app.use(cors({
  origin: 'http://localhost:3002', 
  credentials: true,               
}));

app.use(express.json());

// Routes
const authRoutes = require('./Routes/Authroutes');
app.use('/api/auth', authRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));