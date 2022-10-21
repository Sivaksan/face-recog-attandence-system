-- phpMyAdmin SQL Dump
-- version 5.1.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 11, 2022 at 05:31 AM
-- Server version: 10.4.24-MariaDB
-- PHP Version: 7.4.29

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `database`
--

-- --------------------------------------------------------

--
-- Table structure for table `attendance`
--

CREATE TABLE `attendance` (
  `Name` varchar(50) NOT NULL,
  `Time` time(6) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `attendance`
--

INSERT INTO `attendance` (`Name`, `Time`) VALUES
('Yusuf', '00:00:09.000000'),
('Arham', '00:00:09.300000'),
('Sulax', '00:00:11.000000'),
('chinchan', '00:00:02.000000');

-- --------------------------------------------------------

--
-- Table structure for table `student`
--

CREATE TABLE `student` (
  `fullname` varchar(50) NOT NULL,
  `indexnum` int(10) NOT NULL,
  `email` varchar(100) NOT NULL,
  `degree` varchar(50) NOT NULL,
  `imageupload` varchar(100) NOT NULL,
  `userid` varchar(30) NOT NULL,
  `pwd` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `student`
--

INSERT INTO `student` (`fullname`, `indexnum`, `email`, `degree`, `imageupload`, `userid`, `pwd`) VALUES
('fdf', 43, 'Yusufsalih4@gmail.com', 'se', '', '343', 21232),
('arham', 21212, 'arham@gmail.com', 'se', 'danist-soh-BoYm46b3l6A-unsplash.jpg', '32321', 0),
('user', 21892, 'hath@gmail.com', 'se', 'scan0003.jpg', '2121', 81),
('gre', 78998, 'YYY@gmail.cm', 'cs', 'david-marcu-78A265wPiO4-unsplash.jpg', '454546', 81);

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `email`, `password`) VALUES
(10, 'admin', 'Msyusuf@students.nsbm.ac.lk', '81dc9bdb52d04dc20036dbd8313ed055'),
(11, 'admin', 'admin@gmail.com', '21232f297a57a5a743894a0e4a801fc3');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `student`
--
ALTER TABLE `student`
  ADD PRIMARY KEY (`indexnum`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
