import style from './style.module.css';

export default function LogoNav() {
  return (
    <>
      <div className={style.container}>
        <img src="./images/logo.svg" alt="" />
        <p className={style.title}>Помощник</p>
      </div>
    </>
  );
}
